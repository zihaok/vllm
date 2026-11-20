# ruff: noqa: E501
"""NvGemm2 with a vectorized store (GEMM2's epilogue).

The executor's GEMM2 epilogue (kept by lb24_nvfp4_gemm.NvGemm2) stores each BF16 output element with its own 2-byte store: 8 per thread per
item. The thread's registers hold, per token column, the stored weight rows s, s + 8, s + 16, s + 24 of its warp's 32 (the 16x256b
TMEM load the executor's GEMM1 epilogue relies on: registers 0-1, 2-3, 4-5, 6-7 = those rows, two token columns each), and those are the
unshuffled hidden rows 4 s0 .. 4 s0 + 3 (unshuffled_row): contiguous in C (C is (hidden, rows) with hidden contiguous). So each token
column is one 8-byte store of four BF16 values (cvt.rn.bf16x2.f32: round to nearest even, as the element-wise path). Measured: with
element-wise stores GEMM2 at the served width was slower than TRT-LLM-gen's bmm2 from 8 rows on (11.5 vs 10.0 us at 12 rows, 18.75 vs
14.5 at 24), where a CTA runs 5-14 items. Bitwise equal to NvGemm2.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from . import lb24_nvfp4_gemm as NV

G = NV.G


@dsl_user_op
def store_bf16x4(pointer: cute.Pointer, a, b, c, d, *, loc=None, ip=None):
    """One 8-byte store of four FP32 values rounded to BF16 (nearest even), ``a`` at the lowest address."""
    llvm.inline_asm(
        None,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "{\n.reg .b32 w0, w1;\ncvt.rn.bf16x2.f32 w0, $2, $1;\ncvt.rn.bf16x2.f32 w1, $4, $3;\nst.global.v2.b32 [$0], {w0, w1};\n}",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class NvGemm2V(NV.NvGemm2):
    """NvGemm2 with one 8-byte store per token column (module docstring)."""

    @cute.jit
    def store(
        self, tDrAcc, tDcC, epi, mRoute, cta_words, n_tile, m_tile, item, exchange
    ):
        mC, mAlpha = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        alpha = mAlpha[mRoute[4 + n_tile]]
        c0 = tDcC[0]
        c1 = tDcC[1]
        m0 = G.unshuffled_row(c0[0])
        if c0[1] < row_limit:
            store_bf16x4(
                mC.iterator + (c0[1] * self.m + m0),
                tDrAcc[0] * alpha,
                tDrAcc[2] * alpha,
                tDrAcc[4] * alpha,
                tDrAcc[6] * alpha,
            )
        if c1[1] < row_limit:
            store_bf16x4(
                mC.iterator + (c1[1] * self.m + m0),
                tDrAcc[1] * alpha,
                tDrAcc[3] * alpha,
                tDrAcc[5] * alpha,
                tDrAcc[7] * alpha,
            )
