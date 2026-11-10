# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
import os

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    is_deep_gemm_supported,
    mega_mhc,
)


@functools.cache
def is_mega_mhc_supported(hidden_size: int, hc_mult: int) -> bool:
    """Return whether DeepGEMM can run the DSV4.1 shifted mHC kernel."""
    if not (
        is_deep_gemm_supported()
        and current_platform.is_device_capability_family(100)
        and hidden_size > 0
        and hidden_size % 1024 == 0
        and hc_mult == 4
    ):
        return False
    deep_gemm = _import_deep_gemm()
    return deep_gemm is not None and callable(getattr(deep_gemm, "mega_mhc", None))


@functools.cache
def mega_mhc_supports_mxfp8_y() -> bool:
    """Whether the DeepGEMM build can emit y as FlashInfer-format MXFP8 (y_fi_sf) AND the switch asks for it.

    VLLM_MEGA_MHC_MXFP8_Y DEFAULTS OFF: the MXFP8 y is bit for bit FlashInfer's quant of y, but it
    moves the quantization into a different kernel of the served graph; it is served only when set to 1."""
    if os.environ.get("VLLM_MEGA_MHC_MXFP8_Y", "0") in ("", "0"):
        return False
    deep_gemm = _import_deep_gemm()
    fn = getattr(deep_gemm, "mega_mhc", None) if deep_gemm is not None else None
    try:
        return fn is not None and "y_fi_sf" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def can_use_mega_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    capture_aux: bool,
) -> bool:
    return (
        pre_mix is not None
        and norm_weight is not None
        and not capture_aux
        and x.shape[0] <= 1 << 20
        and is_mega_mhc_supported(x.shape[1], residual.shape[1])
    )


def mhc_shifted_post_pre_deep_gemm(
    x: torch.Tensor,
    residual: torch.Tensor,
    shifted_prev_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    mxfp8_out: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run DSV4.1 shifted post, next pre, and BF16 RMSNorm with Mega mHC.

    With ``mxfp8_out`` (a list), the kernel also writes y as MXFP8 in FlashInfer's
    mxfp8_quantize(is_sf_swizzled_layout=True) format, bit for bit, and appends the
    QuantizedActivation for an MXFP8 consumer that would otherwise quantize y itself.
    """
    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    new_residual = torch.empty_like(residual)
    new_prev_mix = x.new_empty(num_tokens, hc_mult, 1, dtype=torch.float32)
    new_post_mix = torch.empty_like(post_mix)
    new_comb_res_mix = torch.empty_like(comb_res_mix)
    y_bf16 = x.new_empty(num_tokens, hidden_size, dtype=torch.bfloat16)
    mxfp8_kwargs = {}
    if mxfp8_out is not None:
        y_fp8 = torch.empty(
            num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=x.device
        )
        y_fi_sf = torch.empty(
            (num_tokens + 127) // 128 * 128 * (hidden_size // 32),
            dtype=torch.uint8,
            device=x.device,
        )
        mxfp8_kwargs = {"y_fp8": y_fp8, "y_fi_sf": y_fi_sf}
    mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=shifted_prev_mix.unsqueeze(-1),
        post_mix=post_mix,
        comb_res_mix=comb_res_mix,
        fn=fn,
        mix_scales=mix_scales,
        mix_bases=mix_bases,
        hc_mult=hc_mult,
        hc_norm_eps=hc_norm_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_scale=hc_post_scale,
        sinkhorn_eps=sinkhorn_eps,
        num_sinkhorn_iters=num_sinkhorn_iters,
        rmsnorm_weight=rmsnorm_weight,
        rmsnorm_eps=rmsnorm_eps,
        rmsnorm_scale=1.0,
        new_residual=new_residual,
        new_prev_mix=new_prev_mix,
        new_post_mix=new_post_mix,
        new_comb_res_mix=new_comb_res_mix,
        y_bf16=y_bf16,
        **mxfp8_kwargs,
    )
    if mxfp8_out is not None:
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )

        mxfp8_out.append(
            QuantizedActivation(
                mxfp8_kwargs["y_fp8"],
                mxfp8_kwargs["y_fi_sf"],
                y_bf16.dtype,
                y_bf16.shape,
                kMxfp8Dynamic,
            )
        )
    return (
        new_residual,
        new_post_mix,
        new_comb_res_mix,
        y_bf16,
        new_prev_mix.squeeze(-1),
    )
