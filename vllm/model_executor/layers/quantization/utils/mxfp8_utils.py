# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive MXFP8 quantization.
    For each block of 32 elements along the last dimension, compute a
    shared e8m0 scale that fits the block-wise amax into the finite
    float8_e4m3fn range, and quantize each element to float8_e4m3fn.

    Returns (quantized_values [same shape, fp8], scales uint8).
    Scale shape depends on is_sf_swizzled_layout:
      False -> [..., K//32]  (row-major 2D)
      True  -> [flat swizzled 1D]
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    fp8_max = torch.finfo(MXFP8_VALUE_DTYPE).max
    scale_biased = torch.ceil(torch.log2(amax / fp8_max)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        M, K = x.shape
        scales_uint8 = scales_uint8.view(M, -1)
        if is_sf_swizzled_layout:
            scales_uint8 = swizzle_mxfp8_scale(scales_uint8, M=M, K=K)
    elif x.ndim == 3:
        B, M, K = x.shape
        scales_uint8 = scales_uint8.view(B, M, -1)
        if is_sf_swizzled_layout:
            swizzled = []
            for i in range(B):
                swizzled.append(swizzle_mxfp8_scale(scales_uint8[i], M=M, K=K))
            scales_uint8 = torch.cat(swizzled)

    return x_fp8, scales_uint8


def _mxfp8_quant_triton_kernel():
    """Lazily-built Triton kernel: per-32-block E8M0 scale + FP8-E4M3 quant.

    Fuses what ``_mxfp8_e4m3_quantize_torch`` does in several elementwise passes
    into one launch. Each program handles ``[BLOCK_M, 32]`` (one MX block).
    """
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _kernel(
        x_ptr,
        xq_ptr,
        s_ptr,
        M,
        K,
        sxm,
        sxk,
        sqm,
        sqk,
        ssm,
        ssk,
        BLOCK_M: tl.constexpr,
        FP8_MAX: tl.constexpr,
        TINY: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)  # which 32-element block along K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_b * 32 + tl.arange(0, 32)
        m_mask = offs_m < M
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # Mirror _mxfp8_e4m3_quantize_torch: the scale has to put the block amax
        # at the top of the e4m3 range rather than at 1.0, or small elements of
        # the block end up in the subnormals.
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), TINY)  # [BLOCK_M]
        sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        # Scale by the reciprocal rather than dividing by ``exp2(sb - 127)``:
        # sb == 0 (an all-zero / denormal block) makes that divisor 2**-127,
        # which is subnormal in fp32 and flushes to zero on CDNA, turning the
        # block's zeros into 0/0 == NaN. The reciprocal is normal for every
        # reachable sb, and both forms are exact powers of two, so nothing
        # else changes.
        rescale = tl.exp2(127.0 - sb)
        xq = (x * rescale[:, None]).to(xq_ptr.dtype.element_ty)
        tl.store(
            xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
            xq,
            mask=m_mask[:, None],
        )
        tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)

    return _kernel


_MXFP8_QUANT_KERNEL = None


def _mxfp8_e4m3_quantize_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 2D MXFP8 quant (non-swizzled, row-major [M, K//32] scales)."""
    from vllm.triton_utils import triton

    global _MXFP8_QUANT_KERNEL
    if _MXFP8_QUANT_KERNEL is None:
        _MXFP8_QUANT_KERNEL = _mxfp8_quant_triton_kernel()

    M, K = x.shape
    x = x.contiguous()
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (M, K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _MXFP8_QUANT_KERNEL[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
        FP8_MAX=float(torch.finfo(MXFP8_VALUE_DTYPE).max),
        TINY=float(torch.finfo(torch.float32).tiny),
    )
    return xq, scales


# Opt-in Triton kernel for the 2D BF16 -> MXFP8 activation quant with F8_128x4
# swizzled scales (bit-identical to FlashInfer's cute-dsl kernel). FlashInfer's
# row-loop kernel reaches ~5 TB/s on VR200 (DSV4.1 wqa_wkv / wo_b inputs).
_TRITON_SWIZZLED_QUANT = os.environ.get("VLLM_MXFP8_TRITON_QUANT", "0") == "1"
_MXFP8_SWIZZLED_KERNEL = None


def _mxfp8_quant_swizzled_triton_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["M", "padded_m"])
    def _kernel(
        x,
        xq,
        scales,
        M,
        padded_m,
        x_stride,
        K: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        LAUNCH_PDL: tl.constexpr,
    ):
        if LAUNCH_PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        NB: tl.constexpr = BK // 32
        PADDED_COLS: tl.constexpr = K // 32
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BK + tl.arange(0, BK)
        rmask = rows < M
        r64 = rows.to(tl.int64)
        v = tl.load(
            x + r64[:, None] * x_stride + cols[None, :], rmask[:, None], other=0.0
        ).to(tl.float32)
        g = tl.reshape(v, (BM, NB, 32))
        amax = tl.max(tl.abs(g), 2)
        # FlashInfer float_to_ue8m0_fast / ue8m0_to_inv_scale_fast.
        normalized = amax * (1.0 / 448.0)
        bits = normalized.to(tl.uint32, bitcast=True)
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7FFFFF
        bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
        sf = tl.minimum(exponent + bump, 254)
        sf = tl.where(normalized <= 0, 0, sf)
        inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
        inv_scale = inv_bits.to(tl.float32, bitcast=True)
        q = g * inv_scale[:, :, None]
        # Saturate before the satfinite cast like FlashInfer (NaN -> 448).
        q = tl.maximum(tl.minimum(q, 448.0), -448.0)
        q = tl.reshape(q, (BM, BK)).to(tl.float8e4nv)
        tl.store(xq + r64[:, None] * K + cols[None, :], q, rmask[:, None])
        groups = tl.program_id(1) * NB + tl.arange(0, NB)
        sf = tl.where(rmask[:, None], sf, 0).to(tl.uint8)
        # F8_128x4: [row/128, group/4, row%32, row%128/32, group%4].
        offsets = (
            r64[:, None] // 128 * (128 * PADDED_COLS)
            + groups[None, :] // 4 * 512
            + r64[:, None] % 32 * 16
            + r64[:, None] % 128 // 32 * 4
            + groups[None, :] % 4
        )
        tl.store(scales + offsets, sf, rows[:, None] < padded_m)

    return _kernel


def mxfp8_quantize_swizzled_triton(
    x: torch.Tensor, block_m: int = 16, block_k: int = 0, num_warps: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [M, K] -> (e4m3 [M, K], ue8m0 F8_128x4-swizzled scales), K % 128 == 0.

    Defaults (16 x 512 tiles, 4 warps) were the best or within 3% on VR200 for
    K in {1280, 5120, 6144, 8192, 15360} and M in [300, 6144] (~10 TB/s vs ~6 TB/s
    for FlashInfer's cute-dsl kernel; kernels/dense/bench/bench_quant.py)."""
    from vllm.platforms import current_platform
    from vllm.triton_utils import triton

    global _MXFP8_SWIZZLED_KERNEL
    if _MXFP8_SWIZZLED_KERNEL is None:
        _MXFP8_SWIZZLED_KERNEL = _mxfp8_quant_swizzled_triton_kernel()
    M, K = x.shape
    assert K % 128 == 0 and x.stride(1) == 1
    if not block_k:
        block_k = 512 if K % 512 == 0 else (256 if K % 256 == 0 else 128)
    padded_m = (M + 127) // 128 * 128
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        padded_m * (K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    if padded_m:
        launch_pdl = current_platform.is_arch_support_pdl()
        _MXFP8_SWIZZLED_KERNEL[(triton.cdiv(padded_m, block_m), K // block_k)](
            x,
            xq,
            scales,
            M,
            padded_m,
            x.stride(0),
            K=K,
            BM=block_m,
            BK=block_k,
            LAUNCH_PDL=launch_pdl,
            num_warps=num_warps,
            launch_pdl=launch_pdl,
        )
    return xq, scales


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer

    if (
        _TRITON_SWIZZLED_QUANT
        and is_sf_swizzled_layout
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % 128 == 0
        and x.stride(1) == 1
        and alignment in (0, 32)
        and current_platform.is_cuda()
    ):
        return mxfp8_quantize_swizzled_triton(x)

    if current_platform.has_device_capability(100) and has_flashinfer():
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment if alignment > 0 else 32,
            backend="cute-dsl",
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    # ROCm: a single fused Triton kernel beats the multi-pass torch path for the
    # common 2D, non-swizzled activation-quant case (used by the native MX
    # linear/MoE). Falls back to torch otherwise (3D weights, swizzled layout).
    if (
        current_platform.is_rocm()
        and not is_sf_swizzled_layout
        and x.ndim == 2
        and x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    ):
        return _mxfp8_e4m3_quantize_triton(x)

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout, alignment)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
