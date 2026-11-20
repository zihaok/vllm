# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused NVFP4 MoE front of the DeepSeek-V4.1 low-batch routed experts.

The NVFP4 (W4A4) low-batch path (``RoutedExpertGemmNvfp4``) runs three kernels
between the router GEMM and GEMM1: the DSv4 top-k, the modular kernel's input
quantization (``scaled_fp4_quant`` with the layer's static global scale and
linear E4M3 block scales) and the route kernel. :class:`RoutedExpertFrontNvfp4`
is :class:`RoutedExpertFront` (``routed_expert_front_cute.py``) with the
quantizing CTAs writing that NVFP4 layout instead of MXFP8, so the NVFP4 path
gets the MXFP8 path's one-launch front:

* CTA 0: the MXFP8 front's selection and routing record, unchanged;
* CTA ``1 + c``: rows ``3c .. 3c + 2`` of the BF16 input, one thread per
  16-value block (320 per row).

Bitwise contract. Every output equals that of the three kernels it replaces.
The quantization is ``cvt_warp_fp16_to_fp4`` (vLLM ``nvfp4_utils.cuh``,
``scaled_fp4_quant``) instruction by instruction: the block amax in BF16,
``SF = gs * (amax * rcp.approx.ftz(6))`` rounded to E4M3 with saturation, the
output scale ``rcp.approx.ftz(float(SF) * rcp.approx.ftz(gs))`` (0 for a zero
scale), each value times it rounded to E2M1 with saturation, two values a byte
(the even element in the low nibble), the scale byte at ``row * 320 + block``.

PDL as the MXFP8 front: every CTA triggers its dependents on entry; the global
scale (a weight) is read before the grid-dependency wait, the activations after.
"""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import load_global_u32x4

from .routed_expert_front_cute import (
    EXPERTS,
    FRONT_THREADS,
    HIDDEN,
    QUANT_ROWS,
    ROW_THREADS,
    TOP_K,
    RoutedExpertFront,
    abs_max_bf16x2,
    bf16x2_max_to_f32,
    max_bf16x2,
)
from .routed_expert_route_cute import record_words

SF_VEC = 16
NV_BLOCKS = HIDDEN // SF_VEC  # 320 blocks per row, one thread each
assert NV_BLOCKS == ROW_THREADS

# One BF16 pair -> two E2M1 values (even element low) times the inverse scale.
_CVT_PAIR = (
    "and.b32 lo, {w}, 0xFFFF;\nshr.b32 hi, {w}, 16;\nshl.b32 lo, lo, 16;\n"
    "shl.b32 hi, hi, 16;\nmov.b32 f0, lo;\nmov.b32 f1, hi;\n"
    "mul.f32 f0, f0, inv;\nmul.f32 f1, f1, inv;\n"
    "cvt.rn.satfinite.e2m1x2.f32 {b}, f1, f0;\n"
)
# $0, $1: the packed E2M1 bytes of elements 0-7 and 8-15; $2: the E4M3 scale
# byte; $3-$10: the 16 BF16 values as 8 words; $11: the block amax; $12: gs.
_NVFP4_BLOCK16 = (
    "{\n.reg .b32 lo, hi, w;\n.reg .f32 f0, f1, r6, s, q, r, inv;\n"
    ".reg .b16 c, h0, h1;\n.reg .pred p;\n"
    ".reg .b8 y0, y1, y2, y3, y4, y5, y6, y7;\n"
    "mov.f32 r6, 0f40C00000;\nrcp.approx.ftz.f32 r6, r6;\n"
    "mul.f32 s, $11, r6;\nmul.f32 s, $12, s;\n"
    "cvt.rn.satfinite.e4m3x2.f32 c, s, s;\n"
    "cvt.rn.f16x2.e4m3x2 w, c;\nmov.b32 {h0, h1}, w;\ncvt.f32.f16 q, h0;\n"
    "rcp.approx.ftz.f32 r, $12;\nmul.f32 r, q, r;\nrcp.approx.ftz.f32 inv, r;\n"
    "setp.eq.f32 p, q, 0f00000000;\nselp.f32 inv, 0f00000000, inv, p;\n"
    + "".join(_CVT_PAIR.format(w=f"${3 + j}", b=f"y{j}") for j in range(8))
    + "mov.b32 $0, {y0, y1, y2, y3};\nmov.b32 $1, {y4, y5, y6, y7};\n"
    "cvt.u32.u16 $2, c;\nand.b32 $2, $2, 255;\n}"
)


@dsl_user_op
def nvfp4_block16(words, amax, gs, *, loc=None, ip=None):
    """(E2M1 bytes 0-3, bytes 4-7, E4M3 scale byte) of a 16-value BF16 block
    given as 8 words."""
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
def quantize_row_nvfp4(
    x: cute.Tensor,
    row: Int32,
    block: Int32,
    x_quant: cute.Tensor,
    x_scale: cute.Tensor,
    gs: Float32,
):
    """Thread ``block`` of a row quantizes its 16 values to NVFP4."""
    first = row * HIDDEN + block * SF_VEC  # element index
    lo = load_global_u32x4(x.iterator + first // 2)
    hi = load_global_u32x4(x.iterator + first // 2 + 4)
    m01 = max_bf16x2(abs_max_bf16x2(lo[0], lo[1]), abs_max_bf16x2(lo[2], lo[3]))
    m23 = max_bf16x2(abs_max_bf16x2(hi[0], hi[1]), abs_max_bf16x2(hi[2], hi[3]))
    amax = bf16x2_max_to_f32(max_bf16x2(m01, m23))
    v0, v1, code = nvfp4_block16(
        (lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]), amax, gs
    )
    store_global_v2(x_quant.iterator + first // 2, v0, v1)
    x_scale[row * NV_BLOCKS + block] = Uint8(code)


class RoutedExpertFrontNvfp4(RoutedExpertFront):
    """The fused front with NVFP4 quantization (module docstring)."""

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
        """As ``RoutedExpertFront.__call__``, plus ``gs`` (FP32 [1], the
        input's global scale); outputs ``x_quant`` [rows, 2560] E2M1 pairs and
        ``x_scale`` [rows, 320] E4M3 bytes (linear). rows in [1, 8]."""
        slots = cute.make_layout(rows * TOP_K)
        self.kernel_nvfp4(
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
            cute.make_tensor(x_scale_ptr, cute.make_layout(rows * NV_BLOCKS)),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(
                record_ptr, cute.make_layout(record_words(rows, TOP_K, EXPERTS))
            ),
            rows,
        ).launch(
            grid=(1 + (rows + QUANT_ROWS - 1) // QUANT_ROWS, 1, 1),
            block=[FRONT_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel_nvfp4(
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
        """CTA 0 selects and routes (the MXFP8 front's code); the others
        quantize to NVFP4."""
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if cta != 0:
            row = (cta - 1) * QUANT_ROWS + thread // ROW_THREADS
            if thread < QUANT_ROWS * ROW_THREADS and row < rows:
                g = gs[0]
                cute.arch.griddepcontrol_wait()
                quantize_row_nvfp4(x, row, thread % ROW_THREADS, x_quant, x_scale, g)
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
