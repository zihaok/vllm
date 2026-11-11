# MOSCA LB24 (VLLM_MOSCA_LB24, default off): the low-batch routed-expert MoE of the TP4 executor extended from 8 rows
# to 2-3 DSpark requests (12 / 18 rows, up to 24).
#
# Built on the existing low-batch kernels (vllm/models/deepseek_v41/nvidia/specialized_decode/moe/, read-only
# imports; GEMM2 and every helper unchanged). Three pieces are new:
#
#   * RoutedExpertGemm1N: GEMM1 with an N = 16 / 24 (/ 32) column B tile, so one weight pass serves every row of the
#     call. The MMA is the same tcgen05 block-scaled MMA with a wider N; the accumulator is read back 8 columns at a
#     time with exactly the 8-column kernel's TMEM load, and each 8-column group runs the 8-column kernel's epilogue
#     arithmetic unchanged (token = 8 j + column). A group with no token of the tile is skipped.
#   * build_record_multi / RoutedExpertRouteN: the routing record for up to 24 rows. With more than 8 rows an expert
#     can be chosen by more than 8 slots, so it owns ceil(count / 8) consecutive tiles (TRT-LLM-gen's layout: experts
#     ascending, rows ascending, each expert's segment padded to a multiple of 8). Three presence bitmaps (count >= 1,
#     >= 9, >= 17) give an expert's first tile with population counts, as the 8-row kernel does with one.
#   * RoutedExpertFrontN: the fused front (top-k + MXFP8 quantization + record) for up to 24 rows: the 8-row front's
#     selection and merge code, with the (row, part) work units looped over the CTA's 32 warps and the multi-tile
#     record.
#
# Bitwise contract (checked, not assumed): for every token and slot, the GEMM2 row equals the existing
# 8-row pair run on that token's request alone; at <= 8 rows the new route record equals the existing route record.
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

from . import routed_expert_front_cute as F
from . import routed_expert_gemm_cute as G
from . import routed_expert_route_cute as R

TILE = G.TILE
MAX_ROWS_N = 24
ROUTE_N_THREADS = 256  # >= rows * top_k slots (24 x 8 = 192)
WORDS = R.WORDS
MAX_EXPERTS = R.MAX_EXPERTS


# ---------------------------------------------------------------- routing record, multi-tile experts


@cute.jit
def clear_scratch_multi(present: cute.Tensor, present9: cute.Tensor, present17: cute.Tensor, rows_of: cute.Tensor,
                        thread: Int32, threads: cutlass.Constexpr[int]):
    """Zero the three presence bitmaps and the per-expert row masks."""
    for expert in cutlass.range(thread, MAX_EXPERTS, threads, unroll=1):
        rows_of[expert] = Uint32(0)
        if expert < WORDS:
            present[expert] = Uint32(0)
            present9[expert] = Uint32(0)
            present17[expert] = Uint32(0)


@cute.jit
def build_record_multi(present: cute.Tensor, present9: cute.Tensor, present17: cute.Tensor, rows_of: cute.Tensor,
                       ids: cute.Tensor, permuted: cute.Tensor, record: cute.Tensor, rows: Int32, top_k: Int32,
                       experts: Int32, thread: Int32, threads: cutlass.Constexpr[int]):
    """The record of routed_expert_route_cute.build_record for rows <= 24: an expert chosen by c slots owns
    ceil(c / 8) consecutive tiles. Every thread of the block calls it (two block barriers); thread s < rows * top_k
    handles slot s. The record must be zeroed and ordered before the first barrier."""
    slots = rows * top_k
    words = R.tile_words(R.max_tiles(rows, top_k, experts))
    row = thread // top_k
    expert = Int32(-1)
    if thread < slots:
        expert = Int32(ids[thread])
        if expert < 0 or expert >= experts:
            expert = Int32(-1)
    if expert >= 0:
        cute.arch.atomic_or(present.iterator + (expert >> 5), Uint32(1) << Uint32(expert & 31), scope="cta")
        cute.arch.atomic_or(rows_of.iterator + expert, Uint32(1) << Uint32(row), scope="cta")
    cute.arch.sync_threads()
    for e in cutlass.range(thread, experts, threads, unroll=1):
        c = Int32(cute.arch.popc(rows_of[e]))
        if c > TILE:
            cute.arch.atomic_or(present9.iterator + (e >> 5), Uint32(1) << Uint32(e & 31), scope="cta")
        if c > 2 * TILE:
            cute.arch.atomic_or(present17.iterator + (e >> 5), Uint32(1) << Uint32(e & 31), scope="cta")
    cute.arch.sync_threads()
    if thread < slots:
        tiles = Int32(0)
        tile = Int32(0)
        for word in cutlass.range_constexpr(WORDS):
            bits = (Int32(cute.arch.popc(present[word])) + Int32(cute.arch.popc(present9[word]))
                    + Int32(cute.arch.popc(present17[word])))
            tiles += bits
            if word < (expert >> 5):  # never for expert -1
                tile += bits
        if thread == 0:
            record[0] = tiles * TILE
            record[1] = tiles
        if expert < 0:
            permuted[thread] = Int32(-1)
        else:
            w = expert >> 5
            below = (Uint32(1) << Uint32(expert & 31)) - Uint32(1)
            tile += (Int32(cute.arch.popc(present[w] & below)) + Int32(cute.arch.popc(present9[w] & below))
                     + Int32(cute.arch.popc(present17[w] & below)))
            chosen_by = rows_of[expert]
            count = Int32(cute.arch.popc(chosen_by))
            earlier = (Uint32(1) << Uint32(row)) - Uint32(1)
            rank = Int32(cute.arch.popc(chosen_by & earlier))
            first = tile * TILE
            permuted[thread] = first + rank
            record[4 + 2 * words + first + rank] = row
            if rank % TILE == 0:
                t = tile + rank // TILE
                record[4 + t] = expert
                limit = first + count
                if limit > (t + 1) * TILE:
                    limit = (t + 1) * TILE
                record[4 + words + t] = limit


class RoutedExpertRouteN:
    """The route kernel for up to 24 rows (one CTA of 256 threads); PDL as the existing route kernel."""

    @cute.jit
    def __call__(self, ids_ptr: cute.Pointer, permuted_ptr: cute.Pointer, record_ptr: cute.Pointer, rows: Int32,
                 top_k: Int32, experts: Int32, stream: cuda.CUstream):
        slots = cute.make_layout(rows * top_k)
        self.kernel(
            cute.make_tensor(ids_ptr, slots),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(record_ptr, cute.make_layout(R.record_words(rows, top_k, experts))),
            rows, top_k, experts,
        ).launch(grid=(1, 1, 1), block=[ROUTE_N_THREADS, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, ids: cute.Tensor, permuted: cute.Tensor, record: cute.Tensor, rows: Int32, top_k: Int32,
               experts: Int32):
        cute.arch.griddepcontrol_launch_dependents()

        @cute.struct
        class Scratch:
            present: cute.struct.MemRange[Uint32, WORDS]
            present9: cute.struct.MemRange[Uint32, WORDS]
            present17: cute.struct.MemRange[Uint32, WORDS]
            rows_of: cute.struct.MemRange[Uint32, MAX_EXPERTS]

        scratch = cutlass.utils.SmemAllocator().allocate(Scratch)
        present = cute.make_tensor(scratch.present.data_ptr(), cute.make_layout(WORDS))
        present9 = cute.make_tensor(scratch.present9.data_ptr(), cute.make_layout(WORDS))
        present17 = cute.make_tensor(scratch.present17.data_ptr(), cute.make_layout(WORDS))
        rows_of = cute.make_tensor(scratch.rows_of.data_ptr(), cute.make_layout(MAX_EXPERTS))
        thread, _, _ = cute.arch.thread_idx()
        clear_scratch_multi(present, present9, present17, rows_of, thread, ROUTE_N_THREADS)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        R.zero_record(record, rows, top_k, experts, thread, ROUTE_N_THREADS)
        build_record_multi(present, present9, present17, rows_of, ids, permuted, record, rows, top_k, experts, thread,
                           ROUTE_N_THREADS)


# ---------------------------------------------------------------- fused front, up to 24 rows


class RoutedExpertFrontN(F.RoutedExpertFront):
    """The existing fused front with the (row, part) top-k units looped over the 32 warps of CTA 0 and the
    multi-tile record; the quantizing CTAs and every arithmetic helper are the existing ones."""

    @cute.jit
    def select_and_route(self, logits: cute.Tensor, bias: cute.Tensor, bias_vl: cute.Tensor, input_ids: cute.Tensor,
                         ids_are_64bit: Int32, sentinel_lo: Int32, scale: Float32, topk_ids: cute.Tensor,
                         topk_weights: cute.Tensor, permuted: cute.Tensor, record: cute.Tensor, rows: Int32,
                         thread: Int32):
        @cute.struct
        class Shared:
            present: cute.struct.MemRange[Uint32, WORDS]
            present9: cute.struct.MemRange[Uint32, WORDS]
            present17: cute.struct.MemRange[Uint32, WORDS]
            rows_of: cute.struct.MemRange[Uint32, MAX_EXPERTS]
            ids: cute.struct.MemRange[Int32, MAX_ROWS_N * F.TOP_K]
            candidate_key: cute.struct.MemRange[Uint32, MAX_ROWS_N * F.CANDIDATES]
            candidate_id: cute.struct.MemRange[Int32, MAX_ROWS_N * F.CANDIDATES]
            candidate_score: cute.struct.MemRange[Float32, MAX_ROWS_N * F.CANDIDATES]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)

        def view(member, size):
            return cute.make_tensor(member.data_ptr(), cute.make_layout(size))

        present = view(shared.present, WORDS)
        present9 = view(shared.present9, WORDS)
        present17 = view(shared.present17, WORDS)
        rows_of = view(shared.rows_of, MAX_EXPERTS)
        ids_shared = view(shared.ids, MAX_ROWS_N * F.TOP_K)
        candidate_key = view(shared.candidate_key, MAX_ROWS_N * F.CANDIDATES)
        candidate_id = view(shared.candidate_id, MAX_ROWS_N * F.CANDIDATES)
        candidate_score = view(shared.candidate_score, MAX_ROWS_N * F.CANDIDATES)
        clear_scratch_multi(present, present9, present17, rows_of, thread, F.FRONT_THREADS)
        warp = thread >> 5
        lane = thread & 31
        # 32 warps and PARTS = 4: every unit u = warp + 32 i of a warp has part u % 4 = warp % 4.
        part = warp % F.PARTS
        base = part * 32 * F.PER_LANE
        row_bias = cute.make_rmem_tensor((F.PER_LANE,), Float32)
        row_bias_vl = cute.make_rmem_tensor((F.PER_LANE,), Float32)
        for j in cutlass.range_constexpr(F.PER_LANE):
            row_bias[j] = bias[base + lane + 32 * j]
            if cutlass.const_expr(self.image):
                row_bias_vl[j] = bias_vl[base + lane + 32 * j]
        cute.arch.griddepcontrol_wait()
        R.zero_record(record, rows, F.TOP_K, F.EXPERTS, thread, F.FRONT_THREADS)
        for unit in cutlass.range(warp, rows * F.PARTS, 32, unroll=1):
            row = unit // F.PARTS
            logit = cute.make_rmem_tensor((F.PER_LANE,), Float32)
            use_bias = cute.make_rmem_tensor((F.PER_LANE,), Float32)
            for j in cutlass.range_constexpr(F.PER_LANE):
                logit[j] = logits[row * F.EXPERTS + base + lane + 32 * j]
                use_bias[j] = row_bias[j]
            if cutlass.const_expr(self.image):
                token = Int64(input_ids[row])
                if ids_are_64bit != 0:
                    low_word = Int64(Uint32(input_ids[2 * row]))
                    token = (Int64(input_ids[2 * row + 1]) << 32) | low_word
                low = Int64(sentinel_lo)
                if token >= low and token < low + 5:
                    for j in cutlass.range_constexpr(F.PER_LANE):
                        use_bias[j] = row_bias_vl[j]
            key, expert, score = F.part_top_k(logit, use_bias, base, lane)
            if lane < F.TOP_K:
                slot = row * F.CANDIDATES + part * F.TOP_K + lane
                candidate_key[slot] = key
                candidate_id[slot] = expert
                candidate_score[slot] = score
        cute.arch.sync_threads()
        if warp < rows:
            key = Uint32(0)
            expert = Int32(-1)
            score = Float32(0.0)
            if lane < F.CANDIDATES:
                key = Uint32(candidate_key[warp * F.CANDIDATES + lane])
                expert = candidate_id[warp * F.CANDIDATES + lane]
                score = candidate_score[warp * F.CANDIDATES + lane]
            F.merge_row(key, expert, score, scale, warp, lane, topk_ids, topk_weights, ids_shared)
        cute.arch.sync_threads()
        build_record_multi(present, present9, present17, rows_of, ids_shared, permuted, record, rows, Int32(F.TOP_K),
                           Int32(F.EXPERTS), thread, F.FRONT_THREADS)


class RoutedExpertFrontN2(RoutedExpertFrontN):
    """RoutedExpertFrontN with every unit's logits loaded before any selection (a warp holds at most
    ceil(24 x 4 / 32) = 3 units), so the global-load latency is paid once instead of once per pass; the
    selection, merge and record code are unchanged (bitwise the same outputs)."""

    @cute.jit
    def select_and_route(self, logits: cute.Tensor, bias: cute.Tensor, bias_vl: cute.Tensor, input_ids: cute.Tensor,
                         ids_are_64bit: Int32, sentinel_lo: Int32, scale: Float32, topk_ids: cute.Tensor,
                         topk_weights: cute.Tensor, permuted: cute.Tensor, record: cute.Tensor, rows: Int32,
                         thread: Int32):
        @cute.struct
        class Shared:
            present: cute.struct.MemRange[Uint32, WORDS]
            present9: cute.struct.MemRange[Uint32, WORDS]
            present17: cute.struct.MemRange[Uint32, WORDS]
            rows_of: cute.struct.MemRange[Uint32, MAX_EXPERTS]
            ids: cute.struct.MemRange[Int32, MAX_ROWS_N * F.TOP_K]
            candidate_key: cute.struct.MemRange[Uint32, MAX_ROWS_N * F.CANDIDATES]
            candidate_id: cute.struct.MemRange[Int32, MAX_ROWS_N * F.CANDIDATES]
            candidate_score: cute.struct.MemRange[Float32, MAX_ROWS_N * F.CANDIDATES]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)

        def view(member, size):
            return cute.make_tensor(member.data_ptr(), cute.make_layout(size))

        present = view(shared.present, WORDS)
        present9 = view(shared.present9, WORDS)
        present17 = view(shared.present17, WORDS)
        rows_of = view(shared.rows_of, MAX_EXPERTS)
        ids_shared = view(shared.ids, MAX_ROWS_N * F.TOP_K)
        candidate_key = view(shared.candidate_key, MAX_ROWS_N * F.CANDIDATES)
        candidate_id = view(shared.candidate_id, MAX_ROWS_N * F.CANDIDATES)
        candidate_score = view(shared.candidate_score, MAX_ROWS_N * F.CANDIDATES)
        clear_scratch_multi(present, present9, present17, rows_of, thread, F.FRONT_THREADS)
        warp = thread >> 5
        lane = thread & 31
        part = warp % F.PARTS
        base = part * 32 * F.PER_LANE
        row_bias = cute.make_rmem_tensor((F.PER_LANE,), Float32)
        row_bias_vl = cute.make_rmem_tensor((F.PER_LANE,), Float32)
        for j in cutlass.range_constexpr(F.PER_LANE):
            row_bias[j] = bias[base + lane + 32 * j]
            if cutlass.const_expr(self.image):
                row_bias_vl[j] = bias_vl[base + lane + 32 * j]
        cute.arch.griddepcontrol_wait()
        units = rows * F.PARTS
        UNITS = (MAX_ROWS_N * F.PARTS + 31) // 32
        pre = cute.make_rmem_tensor((UNITS * F.PER_LANE,), Float32)
        for p in cutlass.range_constexpr(UNITS):
            unit = warp + 32 * p
            if unit < units:
                row = unit // F.PARTS
                for j in cutlass.range_constexpr(F.PER_LANE):
                    pre[p * F.PER_LANE + j] = logits[row * F.EXPERTS + base + lane + 32 * j]
        R.zero_record(record, rows, F.TOP_K, F.EXPERTS, thread, F.FRONT_THREADS)
        for p in cutlass.range_constexpr(UNITS):
            unit = warp + 32 * p
            if unit < units:
                row = unit // F.PARTS
                logit = cute.make_rmem_tensor((F.PER_LANE,), Float32)
                use_bias = cute.make_rmem_tensor((F.PER_LANE,), Float32)
                for j in cutlass.range_constexpr(F.PER_LANE):
                    logit[j] = pre[p * F.PER_LANE + j]
                    use_bias[j] = row_bias[j]
                if cutlass.const_expr(self.image):
                    token = Int64(input_ids[row])
                    if ids_are_64bit != 0:
                        low_word = Int64(Uint32(input_ids[2 * row]))
                        token = (Int64(input_ids[2 * row + 1]) << 32) | low_word
                    low = Int64(sentinel_lo)
                    if token >= low and token < low + 5:
                        for j in cutlass.range_constexpr(F.PER_LANE):
                            use_bias[j] = row_bias_vl[j]
                key, expert, score = F.part_top_k(logit, use_bias, base, lane)
                if lane < F.TOP_K:
                    slot = row * F.CANDIDATES + part * F.TOP_K + lane
                    candidate_key[slot] = key
                    candidate_id[slot] = expert
                    candidate_score[slot] = score
        cute.arch.sync_threads()
        if warp < rows:
            key = Uint32(0)
            expert = Int32(-1)
            score = Float32(0.0)
            if lane < F.CANDIDATES:
                key = Uint32(candidate_key[warp * F.CANDIDATES + lane])
                expert = candidate_id[warp * F.CANDIDATES + lane]
                score = candidate_score[warp * F.CANDIDATES + lane]
            F.merge_row(key, expert, score, scale, warp, lane, topk_ids, topk_weights, ids_shared)
        cute.arch.sync_threads()
        build_record_multi(present, present9, present17, rows_of, ids_shared, permuted, record, rows, Int32(F.TOP_K),
                           Int32(F.EXPERTS), thread, F.FRONT_THREADS)


# ---------------------------------------------------------------- GEMM1 with an N = 16 / 24 / 32 column B tile


class RoutedExpertGemm1N(G.RoutedExpertGemm1):
    """GEMM1 (the existing RoutedExpertGemm1) with B = all ``rows <= cta_n`` rows of x in one cta_n-column tile."""

    def __init__(self, hidden: int, slice_: int, padded: int, cta_n: int = 16, cta_k: int = 256,
                 num_ab_stage: int = 5) -> None:
        assert cta_n % TILE == 0 and TILE <= cta_n <= 32
        super().__init__(hidden, slice_, padded, cta_k, num_ab_stage)
        self.cta_n = cta_n
        self.groups = cta_n // TILE

    @cute.jit
    def write_b_scales(self, sSFB: cute.Tensor, stage: Int32, k_tile: Int32, b_side: tuple):
        """Rows 0 .. cta_n - 1 of B's scale atoms (rows past x's end get scale 1); the existing layout for row < 32."""
        (b_scale,) = b_side
        atoms = self.cta_k // 128
        stage_bytes = 512 * atoms
        words = cute.make_tensor(cute.recast_ptr(sSFB.iterator, dtype=Uint32),
                                 cute.make_layout(self.num_ab_stage * stage_bytes // 4))
        lane = cute.arch.lane_idx()
        for idx in cutlass.range_constexpr(0, self.cta_n * atoms, 32):
            entry = lane + idx
            if entry < self.cta_n * atoms:
                row = entry // atoms
                atom = entry - row * atoms
                value = Uint32(0x7F7F7F7F)
                if row < cute.size(b_scale, mode=[0]):
                    value = Uint32(b_scale[row, k_tile * atoms + atom])
                words[(stage * stage_bytes + atom * 512 + row * 16) // 4] = value
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma, tma_atom_a: cute.CopyAtom,
               mA_mkl: cute.Tensor, tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom,
               mB_nkl: cute.Tensor, tma_atom_sfb: Optional[cute.CopyAtom], mSFB_nkl: Optional[cute.Tensor],  # noqa: UP045
               mRoute: cute.Tensor, epi: tuple, b_side: tuple,
               cta_words: Int32, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout,
               b_smem_layout_staged: cute.ComposedLayout, sfa_smem_layout_staged: cute.Layout,
               sfb_smem_layout_staged: cute.Layout):
        """The existing persistent kernel (RoutedExpertGemm.kernel) with the epilogue reading the cta_n-column
        accumulator 8 columns at a time; load warps and MMA warp unchanged."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)

        mma_tile_coord_v = Int32(0)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        AB_Stage = self.num_ab_stage
        Acc_Stage = self.num_acc_stage
        EXCHANGE = 2 * self.groups * 4 * TILE

        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, AB_Stage * 2]
            acc_pipeline_bars: cute.struct.MemRange[cutlass.Int64, Acc_Stage * 2]
            tmem_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            # GEMM1's block amax exchange: one buffer per (item parity, column group).
            epi_exchange: cute.struct.MemRange[cutlass.Float32, EXCHANGE]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(), num_stages=AB_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2), consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True)
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_pipeline_bars.data_ptr(), num_stages=Acc_Stage,
            producer_group=CooperativeGroup(Agent.Thread), consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True)
        tmem_bar = storage.tmem_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        sA = smem.allocate_tensor(element_type=self.smem_alloc_a_dtype, layout=a_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(element_type=self.smem_alloc_b_dtype, layout=b_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        sSFA = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128)
        sSFB = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfb_smem_layout_staged, byte_alignment=128)

        # GEMM1 reads the record only after its grid dependency wait (route_before_wait False).
        cute.arch.griddepcontrol_wait()
        total = mRoute[1] * self.m_tiles
        count = Int32(0)
        if bid < total:
            count = (total - 1 - bid) // grid + 1

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        if warp_idx == 0:
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            gA_mkl = cute.local_tile(mA_mkl, (self.cta_m, self.cta_k), (None, None, None))
            tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
                                               cute.group_modes(sA, 0, 3),
                                               cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3))
            gSFA_mkl = cute.local_tile(mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None))
            tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, block_in_cluster_coord_vmnk[2], a_cta_layout,
                                                   cute.group_modes(sSFA, 0, 3),
                                                   cute.group_modes(thr_mma.partition_A(gSFA_mkl), 0, 3))
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            policy = Int64(G.EVICT_FIRST)
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                expert = mRoute[4 + n_tile]
                tAgA_item = tAgA[(None, m_tile, None, expert)]
                tAgSFA_item = tAgSFA[(None, m_tile, None, expert)]
                for k_tile in cutlass.range(self.k_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(ab_full_bar + state.index, self.tma_bytes_a)
                    cute.copy(tma_atom_a, tAgA_item[(None, k_tile)], tAsA[(None, state.index)],
                              tma_bar_ptr=ab_full_bar + state.index, cache_policy=policy)
                    cute.copy(tma_atom_sfa, tAgSFA_item[(None, k_tile)], tAsSFA[(None, state.index)],
                              tma_bar_ptr=ab_full_bar + state.index, cache_policy=policy)
                    state.advance()
            cute.arch.griddepcontrol_launch_dependents()
            for _ in cutlass.range(AB_Stage, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                state.advance()
        elif warp_idx == 1:
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
            tBsB, tBgB = cpasync.tma_partition(tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
                                               cute.group_modes(sB, 0, 3),
                                               cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3))
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            tBgB_item = tBgB[(None, 0, None, 0)]
            for i in cutlass.range(count, unroll=1):
                for k_tile in cutlass.range(self.k_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                    self.write_b_scales(sSFB, state.index, k_tile, b_side)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(ab_full_bar + state.index, self.tma_bytes_b)
                    cute.copy(tma_atom_b, tBgB_item[(None, k_tile)], tBsB[(None, state.index)],
                              tma_bar_ptr=ab_full_bar + state.index)
                    state.advance()
            for _ in cutlass.range(AB_Stage, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                state.advance()
        elif warp_idx == 2:
            self.mma_warp(ab_full_bar, ab_empty_bar, acc_pipeline, tmem_bar, tmem_base_smem_ptr, tiled_mma, sA, sB,
                          sSFA, sSFB, sfa_smem_layout_staged, sfb_smem_layout_staged, count)
        elif warp_idx >= self.epilogue_warp:
            epi_tid = tidx - 32 * self.epilogue_warp
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 0)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
            tCtAcc_base = cute.make_tensor(tmem_ptr, self._acc_layout(tiled_mma))
            # The 8-column kernel's accumulator view, TMEM load and register shape, applied to each 8-column group.
            tiled_mma8 = self._tiled_mma((self.cta_m, TILE))
            acc8_layout = tiled_mma8.make_fragment_C(tiled_mma8.partition_shape_C((self.cta_m, TILE))).layout
            copy_atom_t2r = sm100_utils.get_tmem_load_op((self.cta_m, TILE, self.cta_k),
                                                         cutlass.utils.LayoutEnum.COL_MAJOR, cutlass.BFloat16,
                                                         self.acc_dtype, (self.cta_m, TILE), False)
            acc8_ref = cute.make_tensor(tmem_ptr, acc8_layout)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, acc8_ref[((None, None), 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
            thr_mma8 = tiled_mma8.get_slice(mma_tile_coord_v)
            coord_c = cute.make_identity_tensor((self.m, TILE, 1))
            tCcC = thr_mma8.partition_C(cute.local_tile(coord_c, (self.cta_m, TILE), (None, None, None)))
            tDcC_all = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
            exchange = cute.make_tensor(storage.epi_exchange.data_ptr(), cute.make_layout(EXCHANGE))
            state = make_pipeline_state(PipelineUserType.Consumer, Acc_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                acc_pipeline.consumer_wait(state)
                stage_ptr = tCtAcc_base[(None, None, None, state.index)].iterator
                regs = []
                for j in cutlass.range_constexpr(self.groups):
                    acc_j = cute.make_tensor(stage_ptr + TILE * j, acc8_layout)
                    src = thr_copy_t2r.partition_S(acc_j[((None, None), 0, 0)])
                    reg = cute.make_rmem_tensor(tDcC_all[(None, None, None, 0, 0, 0)].shape, self.acc_dtype)
                    cute.copy(tiled_copy_t2r, src, reg)
                    regs.append(reg)
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(state)
                state.advance()
                for j in cutlass.range_constexpr(self.groups):
                    self.store_group(regs[j], j, epi, mRoute, cta_words, n_tile, m_tile, i, exchange)
                self.store_padding(epi, mRoute, cta_words, n_tile, m_tile)
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.barrier()

    @cute.jit
    def store_group(self, tDrAcc: cute.Tensor, group: cutlass.Constexpr[int], epi: tuple, mRoute: cute.Tensor,
                    cta_words: Int32, n_tile: Int32, m_tile: Int32, item: Int32, exchange: cute.Tensor):
        """The existing GEMM1 epilogue (RoutedExpertGemm1.store) for accumulator columns 8 group .. 8 group + 7,
        i.e. tokens 8 group + 2 (lane % 4) + {0, 1}; skipped when no row of the tile holds a token of the group
        (uniform over the CTA, so the named barrier stays uniform)."""
        mH, mHS, mClamp = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2 + TILE * group
        pair = warp * 16 + (lane // 4) * 2
        limit = mClamp[mRoute[4 + n_tile]]
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        row_a = Int32(-1)
        row_b = Int32(-1)
        hit = Int32(0)
        for r in cutlass.range_constexpr(TILE):
            if r < valid:
                routed = mRoute[4 + 2 * cta_words + row0 + r]
                if routed >= TILE * group and routed < TILE * (group + 1):
                    hit = Int32(1)
                if routed == token:
                    row_a = Int32(r)
                if routed == token + 1:
                    row_b = Int32(r)
        if hit != 0:
            lo_a, lo_b = G.clamped_swiglu_x2(tDrAcc[0], tDrAcc[1], tDrAcc[2], tDrAcc[3], limit, -limit)
            hi_a, hi_b = G.clamped_swiglu_x2(tDrAcc[4], tDrAcc[5], tDrAcc[6], tDrAcc[7], limit, -limit)
            amax_a = G.abs_max_x2(lo_a, hi_a)
            amax_b = G.abs_max_x2(lo_b, hi_b)
            for shift in cutlass.range_constexpr(4, 1, -1):
                amax_a = cute.arch.fmax(amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift), ftz=True)
                amax_b = cute.arch.fmax(amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift), ftz=True)
            base = ((item % 2) * self.groups + group) * (4 * TILE)
            local = token - TILE * group
            if lane < 4:
                exchange[base + warp * TILE + local] = amax_a
                exchange[base + warp * TILE + local + 1] = amax_b
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            partner = base + (warp ^ 1) * TILE + local
            bits_a, bits_b = G.scale_bits_x2(amax_a, exchange[partner], amax_b, exchange[partner + 1])
            code_a, inverse_a = G.ue8m0_code(bits_a)
            code_b, inverse_b = G.ue8m0_code(bits_b)
            h_row = self.padded // 2
            column = (m_tile * 64 + pair) // 2
            if row_a >= 0:
                mH[(row0 + row_a) * h_row + column] = G.quantize_e4m3x2(lo_a, hi_a, inverse_a)
            if row_b >= 0:
                mH[(row0 + row_b) * h_row + column] = G.quantize_e4m3x2(lo_b, hi_b, inverse_b)
            tile_bytes = G.tile_scale_bytes(self.padded)
            if warp % 2 == 0 and lane < TILE:
                block = m_tile * 2 + warp // 2
                row = row_a
                code = code_a
                if lane >= 4:
                    row = row_b
                    code = code_b
                if row >= 0:
                    mHS[n_tile * tile_bytes + (block // 4) * 512 + row * 16 + block % 4] = Uint8(code)

    @cute.jit
    def store_padding(self, epi: tuple, mRoute: cute.Tensor, cta_words: Int32, n_tile: Int32, m_tile: Int32):
        """The existing padding step: the m tile with the slice's last columns zeroes H's padding columns of the
        tile's rows and gives their blocks scale 1."""
        mH, mHS, mClamp = epi
        if m_tile == self.m_tiles - 1:
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
            lane = cute.arch.lane_idx()
            row0 = n_tile * TILE
            valid = mRoute[4 + cta_words + n_tile] - row0
            h_row = self.padded // 2
            tile_bytes = G.tile_scale_bytes(self.padded)
            thread = warp * 32 + lane
            pad_row = thread // 16
            if pad_row < valid:
                pad = (row0 + pad_row) * h_row + (self.slice + (thread % 16) * 4) // 2
                mH[pad] = cutlass.Uint16(0)
                mH[pad + 1] = cutlass.Uint16(0)
            if thread < 2 * TILE:
                pad_row = thread // 2
                pad_block = self.slice // 32 + thread % 2
                if pad_row < valid:
                    mHS[n_tile * tile_bytes + (pad_block // 4) * 512 + pad_row * 16 + pad_block % 4] = Uint8(127)
