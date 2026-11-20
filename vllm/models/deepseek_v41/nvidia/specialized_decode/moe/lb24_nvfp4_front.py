# ruff: noqa: E501
"""The executor's fused low-batch MoE front (routed_expert_front_cute.RoutedExpertFront: DSv4 top-6 + routed-input quantization +
routing record in one launch) with the quantization at NVFP4 and the block scales in the 128x4 layout.

The MXFP4 low-batch path runs front -> GEMM1 -> GEMM2 (3 kernels per layer). At NVFP4 w4a4 the TRT-LLM-gen path runs dsv4_topk ->
scaled_fp4_quant -> routing -> bmm1 -> bmm2 (5 kernels). FrontNV gives the NVFP4 low-batch pair (lb24_nvfp4_gemm) the 3-kernel
shape: CTA 0 selects and routes exactly as the executor's front does (its code, unchanged); the quantizing CTAs write NVFP4.

Quantization (one thread per 16-value block, 320 threads per row, three rows per CTA as the executor's front): the TRT-LLM / vLLM
scaled_fp4_quant sequence -- amax of the block in BF16 (the executor's helpers), SF = E4M3(gs x (amax x 1/6)), inverse =
rcp.approx(float(SF) x rcp.approx(gs)) (0 when the E4M3 scale is 0), values E2M1(x x inverse) (cvt.rn.satfinite.e2m1x2.f32), the scale in
the 128x4 layout scaled_fp4_quant(is_sf_swizzled_layout=True) writes (rows < 8: one atom). Its outputs are bitwise those of
dsv4_topk + scaled_fp4_quant + the route kernel.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import load_global_u32x4

from . import routed_expert_front_cute as F

HIDDEN, TOP_K, EXPERTS = F.HIDDEN, F.TOP_K, F.EXPERTS
NV_BLOCKS = HIDDEN // 16  # 320 blocks per row: one thread each
ROW_THREADS = NV_BLOCKS
QUANT_ROWS = F.FRONT_THREADS // ROW_THREADS
SCALE_BYTES = 128 * NV_BLOCKS  # one 128-row atom set (rows <= 8)
assert ROW_THREADS == F.ROW_THREADS and QUANT_ROWS == F.QUANT_ROWS

_CVT2 = (
    "and.b32 lo, {w}, 0xFFFF;\nshr.b32 hi, {w}, 16;\nshl.b32 lo, lo, 16;\nshl.b32 hi, hi, 16;\n"
    "mov.b32 f0, lo;\nmov.b32 f1, hi;\nmul.f32 f0, f0, inv;\nmul.f32 f1, f1, inv;\n"
    "cvt.rn.satfinite.e2m1x2.f32 {b}, f1, f0;\n"
)
# $0 $1: the packed E2M1 bytes (elements 0-7, 8-15), $2: the E4M3 scale code; $3-$10: the 16 BF16 values as 8 words; $11: amax; $12: gs.
_NVFP4_BLOCK16 = (
    "{\n.reg .b32 lo, hi;\n.reg .f32 f0, f1, s, q, r, inv;\n.reg .b16 c, h0, h1;\n.reg .b32 w;\n.reg .pred p;\n"
    ".reg .b8 y0, y1, y2, y3, y4, y5, y6, y7;\n"
    "mul.rn.f32 s, $11, 0f3E2AAAAB;\nmul.rn.f32 s, $12, s;\n"
    "cvt.rn.satfinite.e4m3x2.f32 c, s, s;\ncvt.rn.f16x2.e4m3x2 w, c;\nmov.b32 {h0, h1}, w;\ncvt.f32.f16 q, h0;\n"
    "rcp.approx.ftz.f32 r, $12;\nmul.rn.f32 r, q, r;\nrcp.approx.ftz.f32 inv, r;\n"
    "setp.eq.f32 p, q, 0f00000000;\nselp.f32 inv, 0f00000000, inv, p;\n"
    + "".join(_CVT2.format(w=f"${3 + j}", b=f"y{j}") for j in range(8))
    + "mov.b32 $0, {y0, y1, y2, y3};\nmov.b32 $1, {y4, y5, y6, y7};\ncvt.u32.u16 $2, c;\nand.b32 $2, $2, 255;\n}"
)


@dsl_user_op
def nvfp4_block16(words, amax, gs, *, loc=None, ip=None):
    """(E2M1 bytes 0-3, bytes 4-7, E4M3 code) of a 16-value BF16 block given as 8 words."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [Uint32(w).ir_value(loc=loc, ip=ip) for w in words]
        + [
            Float32(amax).ir_value(loc=loc, ip=ip),
            Float32(gs).ir_value(loc=loc, ip=ip),
        ],
        _NVFP4_BLOCK16,
        "=r,=r,=r," + ",".join(["r"] * 8) + ",f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(3)
    )


@dsl_user_op
def store_global_v2(pointer: cute.Pointer, a, b, *, loc=None, ip=None):
    """One 8-byte store of two words."""
    llvm.inline_asm(
        None,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def quantize_row_nv(
    x: cute.Tensor,
    row: Int32,
    block: Int32,
    x_quant: cute.Tensor,
    x_scale: cute.Tensor,
    gs: Float32,
):
    """Thread ``block`` of a row quantizes 16 values: amax, E4M3 scale, 8 bytes of E2M1 pairs, the scale in the 128x4 layout."""
    first = row * HIDDEN + block * 16  # element index
    lo = load_global_u32x4(x.iterator + first // 2)
    hi = load_global_u32x4(x.iterator + first // 2 + 4)
    m01 = F.max_bf16x2(F.abs_max_bf16x2(lo[0], lo[1]), F.abs_max_bf16x2(lo[2], lo[3]))
    m23 = F.max_bf16x2(F.abs_max_bf16x2(hi[0], hi[1]), F.abs_max_bf16x2(hi[2], hi[3]))
    amax = F.bf16x2_max_to_f32(F.max_bf16x2(m01, m23))
    v0, v1, code = nvfp4_block16(
        (lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]), amax, gs
    )
    store_global_v2(x_quant.iterator + row * (HIDDEN // 2) + block * 8, v0, v1)
    x_scale[(block // 4) * 512 + row * 16 + block % 4] = cutlass.Uint8(code)


class FrontNV(F.RoutedExpertFront):
    """The executor's fused front with NVFP4 quantization (module docstring)."""

    @cute.jit
    def __call__(
        self,
        logits_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        bias_vl_ptr: cute.Pointer,
        input_ids_ptr: cute.Pointer,
        ids_are_64bit: Int32,
        sentinel_lo: Int32,
        scale: Float32,
        x_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        x_quant_ptr: cute.Pointer,
        x_scale_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        record_ptr: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """As the executor's front, plus ``gs`` (FP32 [1], the input's global scale); outputs ``x_quant`` [rows, 2560] E2M1 pairs and
        ``x_scale`` one 128-row E4M3 atom set (128 x 320 bytes, the 128x4 layout). rows in [1, 8]."""
        slots = cute.make_layout(rows * TOP_K)
        self.kernel_nv(
            cute.make_tensor(logits_ptr, cute.make_layout(rows * EXPERTS)),
            cute.make_tensor(bias_ptr, cute.make_layout(EXPERTS)),
            cute.make_tensor(bias_vl_ptr, cute.make_layout(EXPERTS)),
            cute.make_tensor(input_ids_ptr, cute.make_layout(2 * rows)),
            ids_are_64bit,
            sentinel_lo,
            scale,
            cute.make_tensor(x_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(gs_ptr, cute.make_layout(1)),
            cute.make_tensor(topk_ids_ptr, slots),
            cute.make_tensor(topk_weights_ptr, slots),
            cute.make_tensor(x_quant_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(x_scale_ptr, cute.make_layout(SCALE_BYTES)),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(
                record_ptr, cute.make_layout(F.record_words(rows, TOP_K, EXPERTS))
            ),
            rows,
        ).launch(
            grid=(1 + (rows + QUANT_ROWS - 1) // QUANT_ROWS, 1, 1),
            block=[F.FRONT_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel_nv(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        bias_vl: cute.Tensor,
        input_ids: cute.Tensor,
        ids_are_64bit: Int32,
        sentinel_lo: Int32,
        scale: Float32,
        x: cute.Tensor,
        gs: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        x_quant: cute.Tensor,
        x_scale: cute.Tensor,
        permuted: cute.Tensor,
        record: cute.Tensor,
        rows: Int32,
    ):
        """CTA 0 selects and routes (the executor's code); the others quantize to NVFP4."""
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if cta != 0:
            row = (cta - 1) * QUANT_ROWS + thread // ROW_THREADS
            if thread < QUANT_ROWS * ROW_THREADS and row < rows:
                g = gs[0]
                cute.arch.griddepcontrol_wait()
                quantize_row_nv(x, row, thread % ROW_THREADS, x_quant, x_scale, g)
        else:
            self.select_and_route(
                logits,
                bias,
                bias_vl,
                input_ids,
                ids_are_64bit,
                sentinel_lo,
                scale,
                topk_ids,
                topk_weights,
                permuted,
                record,
                rows,
                thread,
            )
