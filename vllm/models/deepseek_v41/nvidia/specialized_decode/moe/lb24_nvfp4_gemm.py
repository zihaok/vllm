# ruff: noqa: E501
"""The executor's low-batch MoE GEMM pair at NVFP4 w4a4, with x's block scales in the 128x4 layout.

The executor's MXFP4 low-batch pair (routed_expert_gemm_cute.py) runs MXFP4 weights x MXFP8 activations (kind mxf8f6f4, UE8M0
scales per 32). The NVFP4 checkpoint has routed experts in NVFP4 (E2M1 with E4M3 scales per 16 columns and an FP32 global scale).
This module reuses the executor's persistent warp-specialized GEMM unchanged (load warps, MMA warp, staged accumulators, PDL) and
changes only what the dtype decides:

* ``sf_vec_size`` 16 and E4M3 scales, both operands E2M1: the MMA is kind mxf4nvf4 (block-scaled, scale vector 16);
* x's scales are loaded with TMA from one 128-row scale atom (the MXFP8 GEMM1's B load warp wrote UE8M0 words itself);
* FP32 per-expert alphas (the global-scale products) multiply the accumulator before the epilogue;
* GEMM1's epilogue quantizes H to NVFP4: a 16-column block is one warp's 16 columns (8 lanes x 2), so its amax is a 3-step
  butterfly inside the warp (no cross-warp exchange, which the 32-column MXFP8 block needed); scale = E4M3(amax / 6 x gs[e]),
  value = E2M1(h x gs[e] / scale), the TRT-LLM NVFP4 quantization formula with the reciprocals approximate (rcp.approx.ftz);
* H's block scales use the per-tile layout B's scales use (one 128-row atom per 8-row tile, 1024 bytes per 128 columns at 16).

Kernels:
  NvGemm2       GEMM2: BF16 rows alpha2[e] x (H @ w2[e]^T) of each token tile, in the tile's permuted order.
  NvGemm1       GEMM1: H = NVFP4(clamped SwiGLU(alpha1[e] x (x @ w13[e]^T))) into the tile row each token was routed to.
  NvGemm1Plain  GEMM1's MMA path with a BF16 epilogue of every (tile, token) column (diagnostic: the NVFP4 MMA alone).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Uint8, Uint16
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from . import routed_expert_gemm_cute as G

TILE = G.TILE
NV_SF_VEC = 16
E4M3_ONE = 0x38


def nv_tile_scale_bytes(k: int) -> int:
    """Bytes of one 8-row tile's E4M3 block scales over ``k`` columns in the per-tile layout (one 512-byte atom per 64 columns)."""
    return k // 64 * 512


# Two tokens' 16-column blocks (amax a, amax b) and their two values each (lo, hi = columns p, p + 1): the E4M3 scale codes
# (byte 0 = a, byte 1 = b) and the packed E2M1 pairs (byte 0 = a, byte 1 = b; column p in the low nibble).
# scale = E4M3(amax * (1 / 6) * gs); inverse = rcp(rcp(gs) * float(scale)), 0 for an all-zero block; value = E2M1(x * inverse).
_NVFP4_X2 = """{
.reg .pred nq_p;
.reg .f32 nq_sa, nq_sb, nq_qa, nq_qb, nq_r, nq_ia, nq_ib, nq_x0, nq_x1, nq_x2, nq_x3;
.reg .b16 nq_c, nq_f0, nq_f1;
.reg .b32 nq_w;
.reg .b8 nq_ya, nq_yb;
mul.rn.f32 nq_sa, $2, 0f3E2AAAAB;
mul.rn.f32 nq_sa, nq_sa, $8;
mul.rn.f32 nq_sb, $3, 0f3E2AAAAB;
mul.rn.f32 nq_sb, nq_sb, $8;
cvt.rn.satfinite.e4m3x2.f32 nq_c, nq_sb, nq_sa;
cvt.rn.f16x2.e4m3x2 nq_w, nq_c;
mov.b32 {nq_f0, nq_f1}, nq_w;
cvt.f32.f16 nq_qa, nq_f0;
cvt.f32.f16 nq_qb, nq_f1;
rcp.approx.ftz.f32 nq_r, $8;
mul.rn.f32 nq_qa, nq_qa, nq_r;
mul.rn.f32 nq_qb, nq_qb, nq_r;
rcp.approx.ftz.f32 nq_ia, nq_qa;
rcp.approx.ftz.f32 nq_ib, nq_qb;
setp.eq.f32 nq_p, $2, 0f00000000;
selp.f32 nq_ia, 0f00000000, nq_ia, nq_p;
setp.eq.f32 nq_p, $3, 0f00000000;
selp.f32 nq_ib, 0f00000000, nq_ib, nq_p;
mul.rn.f32 nq_x0, $4, nq_ia;
mul.rn.f32 nq_x1, $5, nq_ia;
mul.rn.f32 nq_x2, $6, nq_ib;
mul.rn.f32 nq_x3, $7, nq_ib;
cvt.rn.satfinite.e2m1x2.f32 nq_ya, nq_x1, nq_x0;
cvt.rn.satfinite.e2m1x2.f32 nq_yb, nq_x3, nq_x2;
mov.b16 $0, {nq_ya, nq_yb};
mov.b16 $1, nq_c;
}"""


@dsl_user_op
def nvfp4_x2(amax_a, amax_b, lo_a, hi_a, lo_b, hi_b, gs, *, loc=None, ip=None):
    """(packed E2M1 pairs, E4M3 scale codes) of two tokens' blocks, each a Uint16 with token a in byte 0."""
    result = G._asm(
        llvm.StructType.get_literal([T.i16(), T.i16()]),
        [
            Float32(v).ir_value(loc=loc, ip=ip)
            for v in (amax_a, amax_b, lo_a, hi_a, lo_b, hi_b, gs)
        ],
        _NVFP4_X2,
        "=h,=h,f,f,f,f,f,f,f",
        loc,
        ip,
    )
    return tuple(
        Uint16(llvm.extractvalue(T.i16(), result, [index], loc=loc, ip=ip))
        for index in range(2)
    )


class NvGemm2(G.RoutedExpertGemm2):
    """The executor's GEMM2 at NVFP4 w4a4 with a per-expert FP32 alpha."""

    def __init__(
        self, hidden: int, intermediate: int, cta_k: int = 128, num_ab_stage: int = 10
    ) -> None:
        super().__init__(hidden, intermediate, cta_k, num_ab_stage)
        self.sf_vec_size = NV_SF_VEC

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w2 (hidden, intermediate, experts) E2M1, K-major, E4M3 scales in the 128x4 layout; B: H (rows, intermediate) E2M1,
        ``rows`` the padded rows, scales in the per-tile layout; C: (rows, hidden) BF16; alpha: FP32 per expert."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        padded = cute.ceil_div(rows, TILE) * 128
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((padded, self.k, 1), order=(1, 0, 2)).shape,
                self.sf_vec_size,
            ),
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((self.m, rows), order=(0, 1))
        )
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        self._launch(a, sfa, b, sfb, route, (c, alpha), (), cta_words, ctas, stream)

    @cute.jit
    def store(
        self, tDrAcc, tDcC, epi, mRoute, cta_words, n_tile, m_tile, item, exchange
    ):
        """Alpha x accumulator, rounded to BF16 (nearest even), stored for the tile's rows below its limit at the unshuffled
        hidden index (the executor's GEMM2 epilogue plus the alpha)."""
        mC, mAlpha = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        alpha = mAlpha[mRoute[4 + n_tile]]
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store((tDrAcc.load() * alpha).to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mC[G.unshuffled_row(coord[0]), coord[1]] = tDrC[i]


class _NvGemm1Base(G.RoutedExpertGemm):
    """GEMM1's operands at NVFP4: A w13 (2 x padded, hidden, experts), B the whole routed input x (at most eight rows, rows past
    its end read as zero) with its E4M3 scales loaded by TMA from one 128-row atom."""

    route_before_wait = False

    def __init__(
        self,
        hidden: int,
        slice_: int,
        padded: int,
        cta_k: int = 256,
        num_ab_stage: int = 6,
    ) -> None:
        super().__init__(2 * padded, 2 * slice_ // 128, hidden, cta_k, num_ab_stage)
        self.slice, self.padded = slice_, padded
        self.sf_vec_size = NV_SF_VEC

    @cute.jit
    def _operands(self, a_ptr, sfa_ptr, b_ptr, sfb_ptr, rows, experts):
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((128, self.k, 1), order=(1, 0, 2)).shape,
                self.sf_vec_size,
            ),
        )
        return a, sfa, b, sfb

    @cute.jit
    def b_tile_index(self, n_tile: Int32) -> Int32:
        """Every item multiplies all of ``x``."""
        return Int32(0)


class NvGemm1Plain(_NvGemm1Base):
    """Diagnostic: alpha1[e] x (x @ w13[e]^T) as BF16, column ``8 c + t`` = token ``t`` under token tile ``c``'s expert, row = the
    unshuffled w13 row (up and gate interleaved)."""

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        c_cols: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        a, sfa, b, sfb = self._operands(a_ptr, sfa_ptr, b_ptr, sfb_ptr, rows, experts)
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((self.m, c_cols), order=(0, 1))
        )
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        self._launch(a, sfa, b, sfb, route, (c, alpha), (), cta_words, ctas, stream)

    @cute.jit
    def store(
        self, tDrAcc, tDcC, epi, mRoute, cta_words, n_tile, m_tile, item, exchange
    ):
        mC, mAlpha = epi
        alpha = mAlpha[mRoute[4 + n_tile]]
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store((tDrAcc.load() * alpha).to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            mC[G.unshuffled_row(coord[0]), n_tile * TILE + coord[1]] = tDrC[i]


class NvGemm1(_NvGemm1Base):
    """GEMM1 at NVFP4 (module docstring). The thread holds, for tokens ``t, t + 1`` (``t`` = 2 x lane % 4), the up and gate rows
    of H columns ``p`` (registers 0-3) and ``p + 1`` (4-7), ``p`` = 64 x m_tile + 16 x warp + 2 x lane / 4 (the executor GEMM1's
    mapping); a 16-column block is a warp's 16 columns, lanes with equal lane % 4."""

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        clamp_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """H: (h_rows, padded / 2) bytes of E2M1 pairs; HS: the per-tile layout at 16; clamp / alpha / gs: FP32 per expert."""
        a, sfa, b, sfb = self._operands(a_ptr, sfa_ptr, b_ptr, sfb_ptr, rows, experts)
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.padded // 2)))
        hs = cute.make_tensor(
            hs_ptr,
            cute.make_layout(
                (h_rows + TILE - 1) // TILE * nv_tile_scale_bytes(self.padded)
            ),
        )
        clamp = cute.make_tensor(clamp_ptr, cute.make_layout(experts))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        gs = cute.make_tensor(gs_ptr, cute.make_layout(experts))
        route = cute.make_tensor(
            route_ptr, cute.make_layout(4 + 2 * cta_words + h_rows)
        )
        self._launch(
            a,
            sfa,
            b,
            sfb,
            route,
            (h, hs, clamp, alpha, gs),
            (),
            cta_words,
            ctas,
            stream,
        )

    @cute.jit
    def store(
        self, tDrAcc, tDcC, epi, mRoute, cta_words, n_tile, m_tile, item, exchange
    ):
        mH, mHS, mClamp, mAlpha, mGs = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2
        pair = warp * 16 + (lane // 4) * 2
        expert = mRoute[4 + n_tile]
        limit = mClamp[expert]
        alpha = mAlpha[expert]
        gs = mGs[expert]
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        row_a = Int32(-1)
        row_b = Int32(-1)
        for r in cutlass.range_constexpr(TILE):
            if r < valid:
                routed = mRoute[4 + 2 * cta_words + row0 + r]
                if routed == token:
                    row_a = Int32(r)
                if routed == token + 1:
                    row_b = Int32(r)
        lo_a, lo_b = G.clamped_swiglu_x2(
            tDrAcc[0] * alpha,
            tDrAcc[1] * alpha,
            tDrAcc[2] * alpha,
            tDrAcc[3] * alpha,
            limit,
            -limit,
        )
        hi_a, hi_b = G.clamped_swiglu_x2(
            tDrAcc[4] * alpha,
            tDrAcc[5] * alpha,
            tDrAcc[6] * alpha,
            tDrAcc[7] * alpha,
            limit,
            -limit,
        )
        amax_a = G.abs_max_x2(lo_a, hi_a)
        amax_b = G.abs_max_x2(lo_b, hi_b)
        for shift in cutlass.range_constexpr(4, 1, -1):
            amax_a = cute.arch.fmax(
                amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift), ftz=True
            )
            amax_b = cute.arch.fmax(
                amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift), ftz=True
            )
        values, codes = nvfp4_x2(amax_a, amax_b, lo_a, hi_a, lo_b, hi_b, gs)
        h_row = self.padded // 2
        column = (m_tile * 64 + pair) // 2
        if row_a >= 0:
            mH[(row0 + row_a) * h_row + column] = Uint8(values & Uint16(0xFF))
        if row_b >= 0:
            mH[(row0 + row_b) * h_row + column] = Uint8(values >> Uint16(8))
        tile_bytes = nv_tile_scale_bytes(self.padded)
        if lane < 4:
            # block j = 4 x m_tile + warp: byte (j // 4) x 512 + row x 16 + j % 4 of the tile's atoms
            base = n_tile * tile_bytes + m_tile * 512 + warp
            if row_a >= 0:
                mHS[base + row_a * 16] = Uint8(codes & Uint16(0xFF))
            if row_b >= 0:
                mHS[base + row_b * 16] = Uint8(codes >> Uint16(8))
        if cutlass.const_expr(self.padded > self.slice):  # noqa: SIM102
            if m_tile == self.m_tiles - 1:
                # H's padding columns (none when padded == slice): zero values, scale 1.0 (E4M3 0x38).
                thread = warp * 32 + lane
                pad_row = thread // 16
                if pad_row < valid:
                    pad = (row0 + pad_row) * h_row + self.slice // 2 + (thread % 16) * 2
                    mH[pad] = Uint8(0)
                    mH[pad + 1] = Uint8(0)
                if thread < 4 * TILE:
                    pad_row = thread // 4
                    pad_block = self.slice // 16 + thread % 4
                    if pad_row < valid:
                        mHS[
                            n_tile * tile_bytes
                            + (pad_block // 4) * 512
                            + pad_row * 16
                            + pad_block % 4
                        ] = Uint8(E4M3_ONE)
