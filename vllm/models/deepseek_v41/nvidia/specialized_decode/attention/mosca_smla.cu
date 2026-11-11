// MOSCA sparse-MLA decode kernel for the DeepSeek-V4.1 decode (16 query heads, head dim 512,
// FP8 E4M3 query and KV, one shared 512-wide key/value row per token, BF16 output, per-head attention sink), the drop-in for the
// FlashInfer TRT-LLM-gen call trtllm_batch_decode_sparse_mla_dsv4(sparse_indices_are_storage_offsets=True).
//
// Semantics (the FlashInfer contract for that call): token t of request b (cum_seq_lens_q), column j of sparse_indices[t]:
//   attended iff j < min(sparse_topk_lens[t], W), idx >= 0, and (j >= SWA_WIDTH or j < clamp(seq_lens[b] - (qlen_b - 1 - q_off), 0, SWA_WIDTH));
//   columns < SWA_WIDTH index swa_kv_cache, the rest compressed_kv_cache (storage offsets: row = idx x token stride).
//   s_hj = bmm1_scale * <q_th, k_j>; o_th = bmm2_scale * sum_j softmax_j(s_h., sink_h) k_j   (V = K).
//
// Structure (what this build changes against FlashInfer's MultiCtasKv variant):
//   - one thread-block cluster per token: S CTAs split the token's W columns (S = gridDim.x = cluster size, 8 or 16), so a
//     6-row call runs 6 x S CTAs instead of 96 with a global-partials reduction;
//   - each CTA gathers its KPC keys (cp.async, 16 B per lane) and the token's 16 query rows, QK on FP8 mma.sync (m16n8k32, FP32
//     accumulate), a local softmax in base 2, PV on FP16 mma.sync (m16n8k16, FP32 accumulate; P rounded to FP16, V = K
//     converted exactly from E4M3 to FP16);
//   - the S partial outputs are merged through distributed shared memory: each CTA pushes (remote stores) its partial columns
//     [r * D, r * D + D) and its per-head (max, sum) to rank r, one cluster barrier, then every CTA reduces its own 512 / S output
//     columns of all 16 heads from local shared memory; no global partials, no counters, no second kernel.
// Exactness class: not bitwise to TRT-LLM-gen (different reduction order and P precision); the
// functional rung compares both against an FP64 reference on the same FP8 inputs.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <math.h>

namespace cg = cooperative_groups;

#define NH 16
#define HD 512
#define NWARPS 8
#define NTHREADS (NWARPS * 32)
#define MAX_KPC 512  // chunked path; the single-pass path keeps KPC <= 128
#define MAX_KPC1 128
#define MAX_NT (MAX_KPC1 / 8)
#define OSTR 516  // floats per head row of the partial output (pad 4 against bank conflicts)

struct SmlaParams {
  const uint8_t* q;   long long q_row, q_head;    // bytes
  const uint8_t* swa; long long swa_tok, swa_n;   // bytes per token, tokens in the pool
  const uint8_t* cmp; long long cmp_tok, cmp_n;
  const int* idx;     long long idx_row;          // elements
  const int* lens; const int* seq_lens; const int* cu;
  const float* sinks;                             // [NH] or null
  __nv_bfloat16* out; long long out_row, out_head;  // elements
  float s1_log2;  // bmm1_scale * log2(e)
  float bmm2;
  int W, B, T, KPC, swa_width, pdl, alias, merge, qk_mode, kch;  // merge 2: DSMEM push of FP16 partials  // kch > 0: keys in chunks of kch with an online softmax  // qk_mode 0: 8-key tile per warp, 1: K split over the warps  // merge 0: DSMEM push, 1: global (L2) scratch + cluster barrier
  float* gws; float* gml;
};

#ifdef MOSCA_SMLA_STAMPS
__device__ long long g_stamps[16][12];
#define STAMP(i) do { if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x < 16) g_stamps[blockIdx.x][i] = clock64(); } while (0)
#else
#define STAMP(i) do {} while (0)
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(n) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 16-byte chunk swizzle inside a row: rows 0..7 of an 8x16B ldmatrix tile land in 8 distinct bank groups.
__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ (row & 7); }

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t two) {
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)two, __NV_E4M3);  // exact
  return (uint32_t)hr.x | ((uint32_t)hr.y << 16);
}

static __host__ __device__ inline int kp16_of(int kpc) { return (kpc + 15) & ~15; }
static __host__ __device__ inline size_t align128(size_t x) { return (x + 127) & ~(size_t)127; }

struct SmemLayout {
  size_t q, k, v, p, recv, ml, red, meta, part, sS, total;
};
// recv: the push merge's receive buffer, [S][NH][HD / S] FP32 = 32 KB for any S, plus [S][2 * NH] (m, l) per source CTA.
// alias = 0: recv has its own region; alias = 1: recv overlays K / V and a cluster barrier separates PV from the pushes
// (used when the separate region does not fit the device's shared memory).
static __host__ __device__ inline SmemLayout smem_layout(int kpc, int alias, int qkm = 0, int kch = 0) {
  SmemLayout L; size_t off = 0;
  const int krows = kch > 0 ? 2 * kch : kpc;                  // chunked: two K buffers of kch rows
  const int vrows = kch > 0 ? kch : kp16_of(kpc);
  const int kp16 = vrows;
  const size_t rb = (size_t)NH * HD * 4;
  L.q = off; off += NH * HD;                                   // FP8 Q, swizzled
  L.k = off;
  size_t kv = (size_t)krows * HD + (size_t)vrows * HD * 2;
  L.v = off + (size_t)krows * HD;                              // FP16 V, swizzled (FP8 K before it)
  if (alias) { L.recv = off; off += align128(kv > rb ? kv : rb); }
  else { off += align128(kv); L.recv = off; off += rb; }
  L.ml = off; off += align128((size_t)16 * 2 * NH * 4);        // [S <= 16][m 16, l 16]
  L.p = off; off += align128((size_t)NH * kp16 * 2);           // FP16 P [NH][kp16]
  L.red = off; off += align128((size_t)(MAX_NT * NH * 2 + NH * 16) * 4);  // max/sum per 8-key tile, merge factors
  L.meta = off; off += 16 + MAX_KPC;                            // ints: swa_valid; uint8 valid[MAX_KPC] after
  off = align128(off);
  L.part = off; L.sS = off;
  if (qkm == 1) {                                              // K-split QK: per-warp partial scores, then the reduced scores
    L.part = off; off += align128((size_t)NWARPS * (kpc / 8) * 128 * 4);
    L.sS = off; off += align128((size_t)NH * kpc * 4);
  }
  L.total = off;
  return L;
}

__device__ __forceinline__ uint32_t map_rank(uint32_t local_smem_addr, uint32_t r) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(out) : "r"(local_smem_addr), "r"(r));
  return out;
}
__device__ __forceinline__ void st_cluster_v2(uint32_t addr, float a, float b) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" :: "r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void st_cluster_b32(uint32_t addr, uint32_t a) {
  asm volatile("st.shared::cluster.b32 [%0], %1;\n" :: "r"(addr), "r"(a) : "memory");
}
__device__ __forceinline__ void st_cluster_f32(uint32_t addr, float a) {
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" :: "r"(addr), "f"(a) : "memory");
}
__device__ __forceinline__ void cluster_arrive_release() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait_acquire() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

template <int CHUNKED>
__global__ void __launch_bounds__(NTHREADS, 1) mosca_smla_kernel(const SmlaParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int S = (int)cluster.num_blocks();
  const int rank = (int)cluster.block_rank();
  const int t = blockIdx.y;
  const int KPC = p.KPC, KP16 = p.kch > 0 ? p.kch : kp16_of(KPC), NT = KPC >> 3;
  const SmemLayout Lo = smem_layout(KPC, p.alias, p.qk_mode, p.kch);
  uint8_t* sQ = smem + Lo.q;
  uint8_t* sK = smem + Lo.k;
  uint8_t* sV = smem + Lo.v;
  __half* sP = reinterpret_cast<__half*>(smem + Lo.p);
  float* sRecv = reinterpret_cast<float*>(smem + Lo.recv);  // [S][NH][D]
  float* sML = reinterpret_cast<float*>(smem + Lo.ml);      // [S][2 * NH]
  float* sMaxW = reinterpret_cast<float*>(smem + Lo.red);   // [MAX_NT][NH]
  float* sSumW = sMaxW + MAX_NT * NH;                       // [MAX_NT][NH]
  float* sF = sSumW + MAX_NT * NH;                          // [NH][16]
  const int D = HD / S;
  int* sMeta = reinterpret_cast<int*>(smem + Lo.meta);
  uint8_t* sValid = reinterpret_cast<uint8_t*>(sMeta + 4);  // [MAX_KPC]
  const int j0 = rank * KPC;  // this CTA's first column

  STAMP(0);
  // every CTA of the cluster must have started before any remote shared-memory store reaches it: arrive now, wait before the pushes
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  if (p.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
  STAMP(1);

  const int g = lane >> 2, tq = lane & 3;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
  float my_m = -INFINITY, my_l = 0.f;
  if (CHUNKED == 0) {
  // ---- gather: Q (16 rows) and this CTA's KPC key rows, 16 B per lane (one key per warp per step)
    int meta_cu = 0, meta_seq = 0;
    {
      const int L = min(max(__ldg(p.lens + t), 0), p.W);
      const uint8_t* qrow = p.q + (long long)t * p.q_row;
      for (int c = tid; c < NH * 32; c += NTHREADS) {
        int h = c >> 5, part = c & 31;
        cp_async16(smem_u32(sQ + h * HD + swz(h, part) * 16), qrow + (long long)h * p.q_head + part * 16, true);
      }
      // the request metadata (cu, seq_lens) is loaded by the last warp in the same memory latency as the indices
      if (warp == NWARPS - 1 && p.B <= 31) {
        meta_cu = (lane <= p.B) ? __ldg(p.cu + lane) : 0x7fffffff;
        meta_seq = (lane < p.B) ? __ldg(p.seq_lens + lane) : 0;
      }
      // every index this thread needs is loaded before any copy is issued (one memory latency, not one per step)
      constexpr int MAXIT = MAX_KPC1 * 32 / NTHREADS;
      int ixs[MAXIT];
  #pragma unroll
      for (int it = 0; it < MAXIT; ++it) {
        int c = tid + it * NTHREADS, j = j0 + (c >> 5);
        ixs[it] = (c < KPC * 32 && j < p.W) ? __ldg(p.idx + (long long)t * p.idx_row + j) : -1;
      }
  #pragma unroll
      for (int it = 0; it < MAXIT; ++it) {
        int c = tid + it * NTHREADS;
        if (c >= KPC * 32) break;
        int kk = c >> 5, part = c & 31, j = j0 + kk, ix = ixs[it];
        bool ok = false; const uint8_t* src = p.swa;
        if (j < L) {
          if (j < p.swa_width) { ok = ix >= 0 && ix < p.swa_n; src = p.swa + (long long)ix * p.swa_tok; }
          else { ok = ix >= 0 && ix < p.cmp_n; src = p.cmp + (long long)ix * p.cmp_tok; }
          if (!ok) src = p.swa;
        }
        cp_async16(smem_u32(sK + kk * HD + swz(kk, part) * 16), src + part * 16, ok);
        if (part == 0) sValid[kk] = ok ? 1 : 0;
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    if (p.pdl) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    // request metadata (its loads were issued with the indices): the token's SWA bound
    if (warp == NWARPS - 1) {
      int b = 0, qs, qe, sq;
      if (p.B <= 31) {
        unsigned m = __ballot_sync(0xffffffffu, lane < p.B && meta_cu <= t);
        b = max(0, min(__popc(m) - 1, p.B - 1));
        qs = __shfl_sync(0xffffffffu, meta_cu, b);
        qe = __shfl_sync(0xffffffffu, meta_cu, b + 1);
        sq = __shfl_sync(0xffffffffu, meta_seq, b);
      } else {
        for (int i = 0; i < p.B; ++i) if (__ldg(p.cu + i + 1) > t) { b = i; break; }
        qs = __ldg(p.cu + b); qe = __ldg(p.cu + b + 1); sq = __ldg(p.seq_lens + b);
      }
      if (lane == 0) sMeta[0] = max(0, min(sq - ((qe - qs) - 1 - (t - qs)), p.swa_width));
    }
    // zero the P pad columns (keys KPC..KP16-1) and V pad rows
    for (int c = tid; c < NH * (KP16 - KPC); c += NTHREADS) {
      int h = c / (KP16 - KPC), kk = KPC + c % (KP16 - KPC);
      sP[h * KP16 + kk] = __float2half(0.f);
    }
    for (int c = tid; c < (KP16 - KPC) * 64; c += NTHREADS) {
      int kk = KPC + (c >> 6), part = c & 63;
      *reinterpret_cast<uint4*>(sV + kk * 1024 + part * 16) = make_uint4(0, 0, 0, 0);
    }
    STAMP(2);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    STAMP(3);
  
    // ---- QK: 8-key tile i on warp i % 8 (up to two tiles per warp)
    // K -> FP16 V: key row kk on warp kk % 8 (32 lanes x 16 B), rows of a warp loaded together; warps without a QK tile start at once,
    // the others after their QK
    auto convert_rows = [&]() {
  #pragma unroll 4
      for (int kk = warp; kk < KPC; kk += NWARPS) {
        uint4 w = *reinterpret_cast<const uint4*>(sK + kk * HD + swz(kk, lane) * 16);
        uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        uint32_t o[8];
  #pragma unroll
        for (int i = 0; i < 4; ++i) {
          o[2 * i] = fp8x2_to_f16x2((uint16_t)(ws[i] & 0xffff));
          o[2 * i + 1] = fp8x2_to_f16x2((uint16_t)(ws[i] >> 16));
        }
        *reinterpret_cast<uint4*>(sV + kk * 1024 + swz(kk, 2 * lane) * 16) = make_uint4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<uint4*>(sV + kk * 1024 + swz(kk, 2 * lane + 1) * 16) = make_uint4(o[4], o[5], o[6], o[7]);
      }
    };
    if (p.qk_mode == 1) {
      // K split: warp w takes dims [64w, 64w + 64) (2 k-steps) of every 8-key tile: short independent MMA chains, partials to smem
      float* sPart = reinterpret_cast<float*>(smem + Lo.part);   // [NWARPS][NT][16][8]
      float* sS = reinterpret_cast<float*>(smem + Lo.sS);         // [NH][KPC]
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      uint32_t a0[4], a1[4];
      ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 4 * warp + (lane >> 4)) * 16), a0[0], a0[1], a0[2], a0[3]);
      ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 4 * warp + 2 + (lane >> 4)) * 16), a1[0], a1[1], a1[2], a1[3]);
      for (int tile = 0; tile < NT; ++tile) {
        const int krow = tile * 8 + (lane & 7);
        uint32_t b0, b1, b2, b3;
        ldsm_x4(smem_u32(sK + krow * HD + swz(krow, 4 * warp + (lane >> 3)) * 16), b0, b1, b2, b3);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        mma_fp8(acc, a0[0], a0[1], a0[2], a0[3], b0, b1);
        mma_fp8(acc, a1[0], a1[1], a1[2], a1[3], b2, b3);
        float* dst = sPart + ((size_t)(warp * NT + tile) * 16) * 8;
        *reinterpret_cast<float2*>(dst + g * 8 + 2 * tq) = make_float2(acc[0], acc[1]);
        *reinterpret_cast<float2*>(dst + (g + 8) * 8 + 2 * tq) = make_float2(acc[2], acc[3]);
      }
      convert_rows();
      __syncthreads();
      const int swa_valid = sMeta[0];
      for (int e = tid; e < NH * KPC; e += NTHREADS) {
        int h = e / KPC, kk = e - h * KPC, tile = kk >> 3, col = kk & 7;
        float v = 0.f;
  #pragma unroll
        for (int w = 0; w < NWARPS; ++w) v += sPart[((size_t)(w * NT + tile) * 16 + h) * 8 + col];
        int j = j0 + kk;
        bool ok = sValid[kk] && (j >= p.swa_width || j < swa_valid);
        sS[h * KPC + kk] = ok ? v * p.s1_log2 : -INFINITY;
      }
      __syncthreads();
  #pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int h = warp * 2 + hh;
        float m = -INFINITY;
        for (int kk = lane; kk < KPC; kk += 32) m = fmaxf(m, sS[h * KPC + kk]);
  #pragma unroll
        for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        float sum = 0.f;
        for (int kk = lane; kk < KPC; kk += 32) {
          float v = sS[h * KPC + kk];
          float pr = (v == -INFINITY || m == -INFINITY) ? 0.f : exp2f(v - m);
          sum += pr;
          sP[h * KP16 + kk] = __float2half(pr);
        }
  #pragma unroll
        for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) { sMaxW[h] = m; sSumW[h] = sum; }
      }
    } else {
    float sc[2][4];
      const int my_tiles = (warp < NT) ? ((NT - 1 - warp) / NWARPS + 1) : 0;  // 0, 1 or 2
      {
        const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
          if (ti >= my_tiles) break;
          const int tile = warp + ti * NWARPS;
          float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
          const int krow = tile * 8 + (lane & 7);
    #pragma unroll
          for (int kb = 0; kb < 16; kb += 2) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(smem_u32(sK + krow * HD + swz(krow, 2 * kb + (lane >> 3)) * 16), b0, b1, b2, b3);
            uint32_t a0, a1, a2, a3;
            ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 2 * kb + (lane >> 4)) * 16), a0, a1, a2, a3);
            mma_fp8(acc, a0, a1, a2, a3, b0, b1);
            ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 2 * kb + 2 + (lane >> 4)) * 16), a0, a1, a2, a3);
            mma_fp8(acc2, a0, a1, a2, a3, b2, b3);
          }
    #pragma unroll
          for (int i = 0; i < 4; ++i) sc[ti][i] = acc[i] + acc2[i];
        }
      }
    
      // ---- softmax (base 2) over this CTA's keys
      const int swa_valid = sMeta[0];
    #pragma unroll
      for (int ti = 0; ti < 2; ++ti) {
        if (ti >= my_tiles) break;
        const int tile = warp + ti * NWARPS;
    #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int kk = tile * 8 + 2 * tq + (i & 1);
          int j = j0 + kk;
          bool ok = sValid[kk] && (j >= p.swa_width || j < swa_valid);
          sc[ti][i] = ok ? sc[ti][i] * p.s1_log2 : -INFINITY;
        }
        float m0 = fmaxf(sc[ti][0], sc[ti][1]), m1 = fmaxf(sc[ti][2], sc[ti][3]);
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        if (tq == 0) { sMaxW[tile * NH + g] = m0; sMaxW[tile * NH + g + 8] = m1; }
      }
      convert_rows();
      __syncthreads();
      if (my_tiles > 0) {
        float m0 = -INFINITY, m1 = -INFINITY;
        for (int w = 0; w < NT; ++w) { m0 = fmaxf(m0, sMaxW[w * NH + g]); m1 = fmaxf(m1, sMaxW[w * NH + g + 8]); }
    #pragma unroll
        for (int ti = 0; ti < 2; ++ti) {
          if (ti >= my_tiles) break;
          const int tile = warp + ti * NWARPS;
          float pr[4];
    #pragma unroll
          for (int i = 0; i < 4; ++i) {
            float m = (i < 2) ? m0 : m1;
            pr[i] = (sc[ti][i] == -INFINITY || m == -INFINITY) ? 0.f : exp2f(sc[ti][i] - m);
          }
          float s0 = pr[0] + pr[1], s1 = pr[2] + pr[3];
          s0 += __shfl_xor_sync(0xffffffffu, s0, 1); s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
          s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
          if (tq == 0) { sSumW[tile * NH + g] = s0; sSumW[tile * NH + g + 8] = s1; }
          int kk = tile * 8 + 2 * tq;
          *reinterpret_cast<uint32_t*>(sP + g * KP16 + kk) = pack_f16(pr[0], pr[1]);
          *reinterpret_cast<uint32_t*>(sP + (g + 8) * KP16 + kk) = pack_f16(pr[2], pr[3]);
        }
      }
    }
    __syncthreads();
    STAMP(4);
    my_m = -INFINITY; my_l = 0.f;
    const int NSLOT = (p.qk_mode == 1) ? 1 : NT;  // mode 1 wrote one (max, sum) per head into slot 0
    if (tid < NH) {
      for (int w = 0; w < NSLOT; ++w) my_m = fmaxf(my_m, sMaxW[w * NH + tid]);
      for (int w = 0; w < NSLOT; ++w) my_l += sSumW[w * NH + tid];
    }
  
    // ---- PV: warp w owns output columns [64w, 64w + 64)
    {
      const int prow = (lane & 7) + ((lane >> 3) & 1) * 8;
      for (int kb = 0; kb < KP16; kb += 16) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(smem_u32(sP + prow * KP16 + kb + (lane >> 4) * 8), a0, a1, a2, a3);
        const int vrow = kb + (lane & 7) + ((lane >> 3) & 1) * 8;
  #pragma unroll
        for (int np = 0; np < 4; ++np) {
          int chunk = (warp * 64 + np * 16 + (lane >> 4) * 8) >> 3;
          uint32_t b0, b1, b2, b3;
          ldsm_x4_t(smem_u32(sV + vrow * 1024 + swz(vrow, chunk) * 16), b0, b1, b2, b3);
          mma_f16(o[2 * np], a0, a1, a2, a3, b0, b1);
          mma_f16(o[2 * np + 1], a0, a1, a2, a3, b2, b3);
        }
      }
    }
  } else {
    // ---- chunked: this CTA's KPC columns in chunks of KC rows (double-buffered gather), online softmax per head
    const int KC = p.kch, NTC = KC >> 3, nch = (KPC + KC - 1) / KC;
    float* sRun = sF;  // [0,16) running max, [16,32) running sum, [32,48) rescale of this chunk, [48,64) new max
    const int L = min(max(__ldg(p.lens + t), 0), p.W);
    int meta_cu = 0, meta_seq = 0;
    {
      const uint8_t* qrow = p.q + (long long)t * p.q_row;
      for (int c = tid; c < NH * 32; c += NTHREADS) {
        int h = c >> 5, part = c & 31;
        cp_async16(smem_u32(sQ + h * HD + swz(h, part) * 16), qrow + (long long)h * p.q_head + part * 16, true);
      }
      if (warp == NWARPS - 1 && p.B <= 31) {
        meta_cu = (lane <= p.B) ? __ldg(p.cu + lane) : 0x7fffffff;
        meta_seq = (lane < p.B) ? __ldg(p.seq_lens + lane) : 0;
      }
    }
    auto gather = [&](int c) {
      uint8_t* dst = sK + (size_t)(c & 1) * KC * HD;
      uint8_t* val = sValid + (c & 1) * 64;
      constexpr int MAXC = 64 * 32 / NTHREADS;
      int ixs[MAXC];
#pragma unroll
      for (int it = 0; it < MAXC; ++it) {
        int cc = tid + it * NTHREADS, kk = cc >> 5, jl = c * KC + kk, j = j0 + jl;
        ixs[it] = (cc < KC * 32 && jl < KPC && j < L) ? __ldg(p.idx + (long long)t * p.idx_row + j) : -1;
      }
#pragma unroll
      for (int it = 0; it < MAXC; ++it) {
        int cc = tid + it * NTHREADS;
        if (cc >= KC * 32) break;
        int kk = cc >> 5, part = cc & 31, jl = c * KC + kk, j = j0 + jl, ix = ixs[it];
        bool ok = false; const uint8_t* src = p.swa;
        if (jl < KPC && j < L) {
          if (j < p.swa_width) { ok = ix >= 0 && ix < p.swa_n; src = p.swa + (long long)ix * p.swa_tok; }
          else { ok = ix >= 0 && ix < p.cmp_n; src = p.cmp + (long long)ix * p.cmp_tok; }
          if (!ok) src = p.swa;
        }
        cp_async16(smem_u32(dst + kk * HD + swz(kk, part) * 16), src + part * 16, ok);
        if (part == 0) val[kk] = ok ? 1 : 0;
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    gather(0);
    if (p.pdl) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
    if (warp == NWARPS - 1) {
      int bq = 0, qs, qe, sq;
      if (p.B <= 31) {
        unsigned m = __ballot_sync(0xffffffffu, lane < p.B && meta_cu <= t);
        bq = max(0, min(__popc(m) - 1, p.B - 1));
        qs = __shfl_sync(0xffffffffu, meta_cu, bq);
        qe = __shfl_sync(0xffffffffu, meta_cu, bq + 1);
        sq = __shfl_sync(0xffffffffu, meta_seq, bq);
      } else {
        for (int i = 0; i < p.B; ++i) if (__ldg(p.cu + i + 1) > t) { bq = i; break; }
        qs = __ldg(p.cu + bq); qe = __ldg(p.cu + bq + 1); sq = __ldg(p.seq_lens + bq);
      }
      if (lane == 0) sMeta[0] = max(0, min(sq - ((qe - qs) - 1 - (t - qs)), p.swa_width));
    }
    if (tid < NH) { sRun[tid] = -INFINITY; sRun[16 + tid] = 0.f; }
    STAMP(2);
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    for (int c = 0; c < nch; ++c) {
      const int bb = c & 1;
      if (c + 1 < nch) { gather(c + 1); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();
      const uint8_t* Kb = sK + (size_t)bb * KC * HD;
      const uint8_t* val = sValid + bb * 64;
      const int swa_valid = sMeta[0];
      float sc4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      if (warp < NTC) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
        const int krow = warp * 8 + (lane & 7);
#pragma unroll
        for (int kb = 0; kb < 16; kb += 2) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4(smem_u32(Kb + krow * HD + swz(krow, 2 * kb + (lane >> 3)) * 16), b0, b1, b2, b3);
          uint32_t a0, a1, a2, a3;
          ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 2 * kb + (lane >> 4)) * 16), a0, a1, a2, a3);
          mma_fp8(acc, a0, a1, a2, a3, b0, b1);
          ldsm_x4(smem_u32(sQ + arow * HD + swz(arow, 2 * kb + 2 + (lane >> 4)) * 16), a0, a1, a2, a3);
          mma_fp8(acc2, a0, a1, a2, a3, b2, b3);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int kk = warp * 8 + 2 * tq + (i & 1);
          int j = j0 + c * KC + kk;
          bool ok = val[kk] && (j >= p.swa_width || j < swa_valid);
          sc4[i] = ok ? (acc[i] + acc2[i]) * p.s1_log2 : -INFINITY;
        }
        float m0 = fmaxf(sc4[0], sc4[1]), m1 = fmaxf(sc4[2], sc4[3]);
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        if (tq == 0) { sMaxW[warp * NH + g] = m0; sMaxW[warp * NH + g + 8] = m1; }
      }
#pragma unroll 4
      for (int kk = warp; kk < KC; kk += NWARPS) {
        uint4 w = *reinterpret_cast<const uint4*>(Kb + kk * HD + swz(kk, lane) * 16);
        uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        uint32_t ov[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ov[2 * i] = fp8x2_to_f16x2((uint16_t)(ws[i] & 0xffff));
          ov[2 * i + 1] = fp8x2_to_f16x2((uint16_t)(ws[i] >> 16));
        }
        *reinterpret_cast<uint4*>(sV + kk * 1024 + swz(kk, 2 * lane) * 16) = make_uint4(ov[0], ov[1], ov[2], ov[3]);
        *reinterpret_cast<uint4*>(sV + kk * 1024 + swz(kk, 2 * lane + 1) * 16) = make_uint4(ov[4], ov[5], ov[6], ov[7]);
      }
      __syncthreads();
      if (warp < NTC) {
        float m0 = sRun[g], m1 = sRun[g + 8];
        for (int w = 0; w < NTC; ++w) { m0 = fmaxf(m0, sMaxW[w * NH + g]); m1 = fmaxf(m1, sMaxW[w * NH + g + 8]); }
        float pr[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float m = (i < 2) ? m0 : m1;
          pr[i] = (sc4[i] == -INFINITY || m == -INFINITY) ? 0.f : exp2f(sc4[i] - m);
        }
        float s0 = pr[0] + pr[1], s1 = pr[2] + pr[3];
        s0 += __shfl_xor_sync(0xffffffffu, s0, 1); s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        if (tq == 0) { sSumW[warp * NH + g] = s0; sSumW[warp * NH + g + 8] = s1; }
        int kk = warp * 8 + 2 * tq;
        *reinterpret_cast<uint32_t*>(sP + g * KC + kk) = pack_f16(pr[0], pr[1]);
        *reinterpret_cast<uint32_t*>(sP + (g + 8) * KC + kk) = pack_f16(pr[2], pr[3]);
      }
      if (tid < NH) {
        float mo = sRun[tid], mn = mo;
        for (int w = 0; w < NTC; ++w) mn = fmaxf(mn, sMaxW[w * NH + tid]);
        sRun[32 + tid] = (mo == -INFINITY) ? 0.f : exp2f(mo - mn);
        sRun[48 + tid] = mn;
      }
      __syncthreads();
      if (tid < NH) {
        float sum = 0.f;
        for (int w = 0; w < NTC; ++w) sum += sSumW[w * NH + tid];
        sRun[16 + tid] = sRun[16 + tid] * sRun[32 + tid] + sum;
        sRun[tid] = sRun[48 + tid];
      }
      const float a0r = sRun[32 + g], a1r = sRun[32 + g + 8];
#pragma unroll
      for (int n = 0; n < 8; ++n) { o[n][0] *= a0r; o[n][1] *= a0r; o[n][2] *= a1r; o[n][3] *= a1r; }
      {
        const int prow = (lane & 7) + ((lane >> 3) & 1) * 8;
        for (int kb = 0; kb < KC; kb += 16) {
          uint32_t a0, a1, a2, a3;
          ldsm_x4(smem_u32(sP + prow * KC + kb + (lane >> 4) * 8), a0, a1, a2, a3);
          const int vrow = kb + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
          for (int np = 0; np < 4; ++np) {
            int chunk = (warp * 64 + np * 16 + (lane >> 4) * 8) >> 3;
            uint32_t b0, b1, b2, b3;
            ldsm_x4_t(smem_u32(sV + vrow * 1024 + swz(vrow, chunk) * 16), b0, b1, b2, b3);
            mma_f16(o[2 * np], a0, a1, a2, a3, b0, b1);
            mma_f16(o[2 * np + 1], a0, a1, a2, a3, b2, b3);
          }
        }
      }
      __syncthreads();
    }
    STAMP(3);
    STAMP(4);
    if (tid < NH) { my_m = sRun[tid]; my_l = sRun[16 + tid]; }
  }
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");  // the start barrier: every CTA has started
  if (p.alias) {  // recv overlays K / V: every CTA of the cluster has finished PV before anyone pushes
    cluster_arrive_release();
    cluster_wait_acquire();
  }
  STAMP(5);
  // push: dims [r * D, r * D + D) of this CTA's partial go to rank r (recv[rank][h][d - r * D]), with this CTA's per-head (m, l)
  if (p.merge == 2) {  // DSMEM push of FP16 partials: half the bytes of merge 0 (|partial| <= KPC x 448 fits FP16)
    const uint32_t recv0 = smem_u32(sRecv);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      int d = warp * 64 + n * 8 + 2 * tq;
      int r = d / D, dl = d - r * D;
      uint32_t base = map_rank(recv0, (uint32_t)r) + (uint32_t)((rank * NH * D + dl) * 2);
      st_cluster_b32(base + (uint32_t)(g * D * 2), pack_f16(o[n][0], o[n][1]));
      st_cluster_b32(base + (uint32_t)((g + 8) * D * 2), pack_f16(o[n][2], o[n][3]));
    }
    if (tid < NH) {
      const uint32_t ml0 = smem_u32(sML + rank * 2 * NH);
      for (int r = 0; r < S; ++r) {
        uint32_t a = map_rank(ml0, (uint32_t)r);
        st_cluster_f32(a + tid * 4, my_m);
        st_cluster_f32(a + (NH + tid) * 4, my_l);
      }
    }
  } else if (p.merge == 0) {
    const uint32_t recv0 = smem_u32(sRecv);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      int d = warp * 64 + n * 8 + 2 * tq;
      int r = d / D, dl = d - r * D;
      uint32_t base = map_rank(recv0, (uint32_t)r) + (uint32_t)((rank * NH * D + dl) * 4);
      st_cluster_v2(base + (uint32_t)(g * D * 4), o[n][0], o[n][1]);
      st_cluster_v2(base + (uint32_t)((g + 8) * D * 4), o[n][2], o[n][3]);
    }
    if (tid < NH) {
      const uint32_t ml0 = smem_u32(sML + rank * 2 * NH);
      for (int r = 0; r < S; ++r) {
        uint32_t a = map_rank(ml0, (uint32_t)r);
        st_cluster_f32(a + tid * 4, my_m);
        st_cluster_f32(a + (NH + tid) * 4, my_l);
      }
    }
  } else {
    float* wsT = p.gws + (size_t)t * S * NH * HD;  // [owner r][src][NH][D]
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      int d = warp * 64 + n * 8 + 2 * tq;
      int r = d / D, dl = d - r * D;
      float* base = wsT + ((size_t)(r * S + rank) * NH) * D + dl;
      __stcg(reinterpret_cast<float2*>(base + g * D), make_float2(o[n][0], o[n][1]));
      __stcg(reinterpret_cast<float2*>(base + (g + 8) * D), make_float2(o[n][2], o[n][3]));
    }
    if (tid < NH) {
      __stcg(p.gml + ((size_t)t * S + rank) * 2 * NH + tid, my_m);
      __stcg(p.gml + ((size_t)t * S + rank) * 2 * NH + NH + tid, my_l);
    }
  }
  cluster_arrive_release();
  cluster_wait_acquire();
  STAMP(6);
  if (p.merge == 1) {  // stage this CTA's slice from L2 into recv / ML (same layout as the DSMEM path)
    const float* src = p.gws + (size_t)t * S * NH * HD + (size_t)rank * S * NH * D;
    for (int c = tid; c < S * NH * D / 4; c += NTHREADS)
      reinterpret_cast<float4*>(sRecv)[c] = __ldcg(reinterpret_cast<const float4*>(src) + c);
    for (int c = tid; c < S * 2 * NH; c += NTHREADS) sML[c] = __ldcg(p.gml + (size_t)t * S * 2 * NH + c);
    __syncthreads();
  }

  // ---- local reduce of this CTA's columns [rank * D, rank * D + D) over the S partials
  if (tid < NH * S) {
    int h = tid / S, r = tid % S;
    float mr = sML[r * 2 * NH + h], lr = sML[r * 2 * NH + NH + h];
    float M = mr;
    for (int off = 1; off < S; off <<= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, off));
    float sink2 = p.sinks ? __ldg(p.sinks + h) * 1.4426950408889634f : -INFINITY;
    M = fmaxf(M, sink2);
    float e = (mr == -INFINITY || M == -INFINITY) ? 0.f : exp2f(mr - M);
    float Lsum = lr * e;
    for (int off = 1; off < S; off <<= 1) Lsum += __shfl_xor_sync(0xffffffffu, Lsum, off);
    if (sink2 != -INFINITY) Lsum += exp2f(sink2 - M);
    sF[h * 16 + r] = (Lsum > 0.f) ? e * p.bmm2 / Lsum : 0.f;
  }
  __syncthreads();
  {
    const int half = D / 2;
    for (int k = tid; k < NH * half; k += NTHREADS) {
      int h = k / half, dl = 2 * (k % half);
      float2 acc2 = make_float2(0.f, 0.f);
      for (int r = 0; r < S; ++r) {
        float f = sF[h * 16 + r];
        float2 v = (p.merge == 2) ? __half22float2(*reinterpret_cast<const __half2*>(reinterpret_cast<const __half*>(sRecv) + (r * NH + h) * D + dl))
                                  : *reinterpret_cast<const float2*>(sRecv + (r * NH + h) * D + dl);
        acc2.x += f * v.x; acc2.y += f * v.y;
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (long long)t * p.out_row + (long long)h * p.out_head + rank * D + dl) =
          __floats2bfloat162_rn(acc2.x, acc2.y);
    }
  }
  STAMP(7);
}

extern "C" {

// Returns 0 on success, otherwise a cudaError_t (or -1 for a bad configuration).
int mosca_smla_launch(const void* q, long long q_row, long long q_head,
                      const void* swa, long long swa_tok, long long swa_n,
                      const void* cmp, long long cmp_tok, long long cmp_n,
                      const int* idx, long long idx_row, int W,
                      const int* lens, const int* seq_lens, const int* cu, int B,
                      const float* sinks, float bmm1_scale, float bmm2_scale,
                      void* out, long long out_row, long long out_head,
                      int T, int S, int KPC, int swa_width, int pdl, int merge, void* gws, void* gml, int qk_mode, int kch, void* stream) {
  if (T <= 0) return 0;
  if (S < 2 || S > 16 || (S & (S - 1)) || KPC < 8 || KPC > MAX_KPC || (KPC & 7) || S * KPC < W) return -1;
  if (kch == 0 && KPC > MAX_KPC1) return -1;
  if (kch != 0 && (kch < 16 || kch > 64 || (kch & 15) || qk_mode != 0)) return -1;
  if (HD % S) return -1;
  SmlaParams p;
  p.q = (const uint8_t*)q; p.q_row = q_row; p.q_head = q_head;
  p.swa = (const uint8_t*)swa; p.swa_tok = swa_tok; p.swa_n = swa_n;
  p.cmp = (const uint8_t*)cmp; p.cmp_tok = cmp_tok; p.cmp_n = cmp_n;
  p.idx = idx; p.idx_row = idx_row; p.W = W;
  p.lens = lens; p.seq_lens = seq_lens; p.cu = cu; p.B = B;
  p.sinks = sinks; p.s1_log2 = bmm1_scale * 1.4426950408889634f; p.bmm2 = bmm2_scale;
  p.out = (__nv_bfloat16*)out; p.out_row = out_row; p.out_head = out_head;
  p.T = T; p.KPC = KPC; p.swa_width = swa_width; p.pdl = pdl;
  p.merge = merge; p.gws = (float*)gws; p.gml = (float*)gml; p.qk_mode = qk_mode ? 1 : 0; p.kch = kch;
  if (merge == 1 && (!gws || !gml)) return -1;
  if (merge < 0 || merge > 2) return -1;
  static int configured = 0, optin = 0;
  if (!configured) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaError_t e = cudaFuncSetAttribute(mosca_smla_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(mosca_smla_kernel<0>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(mosca_smla_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
    if (e != cudaSuccess) return (int)e;
    e = cudaFuncSetAttribute(mosca_smla_kernel<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
    configured = 1;
  }
  p.alias = smem_layout(KPC, 0, p.qk_mode, kch).total > (size_t)optin ? 1 : 0;
  SmemLayout L = smem_layout(KPC, p.alias, p.qk_mode, kch);
  if (L.total > (size_t)optin) return -1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(S, T, 1);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = L.total;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attr; cfg.numAttrs = 2;
  cudaError_t e = kch ? cudaLaunchKernelEx(&cfg, mosca_smla_kernel<1>, p) : cudaLaunchKernelEx(&cfg, mosca_smla_kernel<0>, p);
  return (int)e;
}

// The most clusters of size S that can be co-resident (0 on error), for the split table's sizing.
int mosca_smla_max_active_clusters(int S, int KPC) {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  SmemLayout L = smem_layout(KPC, smem_layout(KPC, 0).total > (size_t)optin ? 1 : 0);
  cudaFuncSetAttribute(mosca_smla_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  cudaFuncSetAttribute(mosca_smla_kernel<0>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(S, 64, 1); cfg.blockDim = dim3(NTHREADS, 1, 1); cfg.dynamicSmemBytes = L.total;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, (void*)mosca_smla_kernel<0>, &cfg) != cudaSuccess) return 0;
  return n;
}

// The shared memory a launch uses (the separate receive buffer when it fits the device, else the aliased layout).
int mosca_smla_smem_bytes(int KPC) {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  size_t a = smem_layout(KPC, 0).total;
  return (int)(a > (size_t)optin ? smem_layout(KPC, 1).total : a);
}
int mosca_smla_uses_alias(int KPC) {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  return smem_layout(KPC, 0).total > (size_t)optin ? 1 : 0;
}

int mosca_smla_read_stamps(long long* host) {
#ifdef MOSCA_SMLA_STAMPS
  return (int)cudaMemcpyFromSymbol(host, g_stamps, sizeof(long long) * 16 * 12);
#else
  (void)host; return -2;
#endif
}

}  // extern "C"
