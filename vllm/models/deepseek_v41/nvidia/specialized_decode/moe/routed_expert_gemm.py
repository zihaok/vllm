# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-batch routed-expert GEMMs for DeepSeek-V4.1 decode (1-8 rows).

At concurrency 1 with DSpark, every target MoE layer applies its routed experts
to six rows (384 experts, top-6; at TP4 every rank holds all experts at
intermediate 576, padded to 640). ``TrtLlmMxfp4ExpertsModular`` runs FlashInfer's
TRT-LLM-gen routing kernel and two batched MXFP8 x MXFP4 GEMMs; at this size they
stream the weights of about 22 experts per layer, so their cost is HBM time plus
fixed per-kernel latency. This module computes the same values from the same
tensors with CuTe DSL kernels sized for at most eight rows::

    router logits ─► front ───────────────────────┐ (top-k, MXFP8 input, routing)
    top-k ids, MXFP8 input ─► route ──────────────┤
                                                  ▼
                         GEMM1 ─► h (MXFP8) ─► GEMM2 ─► unweighted permuted rows
                                                          │
       UnfinalizedMoEOutput (fused all-reduce + mHC) ◄────┘

* ``routed_expert_route_cute.py`` builds the expert-major, 8-row-tile
  permutation and the per-tile record both GEMMs read, as FlashInfer's routing
  kernel does with precomputed ids and an 8-token tile.
* ``routed_expert_gemm_cute.py`` holds GEMM1 and GEMM2: persistent 128 x 8
  ``tcgen05`` block-scaled tiles over the (weight tile, token tile) work items
  of the record, with evict-first hints on the expert weights. GEMM1 skips the
  zero padding tile of the 576-wide slice. Their numerics are those of the
  TRT-LLM-gen ``t128x8`` tactics.
* ``routed_expert_front_cute.py`` replaces the DSv4 top-k (Triton), the MXFP8
  quantization of the routed input and the route kernel with one launch whose
  outputs are bitwise those of the three; it is built for the target's 384
  experts and top-6.

The kernels read the tensors vLLM prepares for the TRT-LLM-gen MXFP4 backend:
``w13`` [E, 1280, 2560] u8 (interleaved up/gate rows shuffled for 128-row
epilogue tiles) with [E, 1280, 160] scales in the 128x4 layout, ``w2``
[E, 5120, 320] u8 with [E, 5120, 20] scales, and the per-expert SwiGLU clamp.
Each kernel is compiled once per device when the layers are bound (shapes,
expert counts and the routing record are launch arguments) and launched on the
current stream with programmatic dependent launch, so CUDA graphs capture
them like any other kernel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .routed_expert_front_cute import RoutedExpertFront
from .routed_expert_front_nvfp4_cute import RoutedExpertFrontNvfp4
from .routed_expert_gemm_cute import (
    RoutedExpertGemm1,
    RoutedExpertGemm2,
    tile_scale_bytes,
)
from .routed_expert_gemm_nvfp4_cute import (
    SF_VEC as NVFP4_SF_VEC,
)
from .routed_expert_gemm_nvfp4_cute import (
    RoutedExpertGemm1Nvfp4,
    RoutedExpertGemm2Nvfp4,
    tile_scale_bytes_nvfp4,
)
from .routed_expert_route_cute import RoutedExpertRoute

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEPrepareAndFinalizeModular,
    )

logger = init_logger(__name__)

HIDDEN = 5120
SLICE = 576
PADDED = 640
# The NVFP4 experts' intermediate at TP4: vLLM pads a gated NVFP4 intermediate
# to a multiple of 64 only, so 576 stays unpadded.
NV_INTERMEDIATE = 576
# The fused front is built for exactly the target layers' routing.
EXPERTS = 384
TOP_K = 6
# The route kernel and the GEMMs take any multiple of 32 local experts up to
# MAX_EXPERTS, and any top-k whose 8 rows fit the route kernel.
MAX_EXPERTS = 384
MAX_TOP_K = 8
TILE = 8
MAX_ROWS = 8


def max_ctas(rows: int, top_k: int, experts: int) -> int:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = min(experts, slots)
    return filled + (slots - filled) // TILE


def cta_words(ctas: int) -> int:
    """``ctas`` rounded up to a multiple of four (the record's table width)."""
    return (ctas + 3) // 4 * 4


def route_words(ctas: int) -> int:
    """int32 length of the routing record for ``ctas`` token tiles."""
    return 4 + 2 * cta_words(ctas) + TILE * ctas


@dataclass
class FrontConfig:
    """The router's DSv4 top-k parameters the fused front applies."""

    bias: torch.Tensor  # FP32 [384] score correction bias
    bias_vl: torch.Tensor | None  # FP32 [384] image-token bias, None without
    image_sentinel_lo: int  # first of the five image sentinel token ids
    scale: float  # routed scaling factor of the normalized weights


@dataclass
class Routing:
    """What GEMM1 and GEMM2 read, from ``route`` or ``front``."""

    x_quant: torch.Tensor  # MXFP8 activations [rows, 5120]
    x_scale: torch.Tensor  # their UE8M0 scales, linear layout
    topk_ids: torch.Tensor  # int32 [rows, top_k]
    topk_weights: torch.Tensor  # FP32 [rows, top_k]
    permuted: torch.Tensor  # int32 [rows, top_k], slot -> permuted row
    record: torch.Tensor  # int32 routing record (routed_expert_route_cute.py)
    ctas: int  # token tiles the GEMMs' records allow for


class RoutedExpertGemm:
    """The route kernel (or fused front) and GEMM pair bound to one MoE
    layer's TRT-LLM-gen MXFP4 experts (see the module docstring)."""

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        clamp: torch.Tensor,
        top_k: int,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        quant_config: FusedMoEQuantConfig,
    ):
        check_weights(w13, w13_scale, w2, w2_scale, clamp, top_k)
        self.experts = int(w13.size(0))
        self.top_k = top_k
        index = w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        # Compile the kernels now, not inside a CUDA graph capture.
        _route_kernel(self.device_index)
        _gemm1_kernel(self.device_index)
        _gemm2_kernel(self.device_index)
        # The modular kernel's input quantization, reused so the activations
        # match its path bit for bit.
        self.prepare_finalize = prepare_finalize
        self.quant_config = quant_config
        self.front_config: FrontConfig | None = None
        self.w13, self.w13_scale, self.w2, self.w2_scale = w13, w13_scale, w2, w2_scale
        self.clamp = clamp

    def set_front(self, config: FrontConfig) -> None:
        """Use the fused front with the router's ``config``."""
        _front_kernel(self.device_index, config.bias_vl is not None)
        self.front_config = config

    def accepts(self, x: torch.Tensor, runner: torch.nn.Module) -> bool:
        """Whether a call with router input ``x`` can take these kernels: 1-8
        BF16 rows, and the runner's expert weights still the tensors bound here
        (a weight reload replaces them)."""
        return (
            1 <= x.size(0) <= MAX_ROWS
            and x.dtype == torch.bfloat16
            and tuple(x.shape) == (x.size(0), HIDDEN)
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and runner.routed_experts.w13_weight.data_ptr() == self.w13.data_ptr()
        )

    def route(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Routing:
        """Quantize ``x`` as the modular kernel does and launch the route kernel
        on the router's top-k (``topk_weights`` FP32)."""
        rows, top_k = topk_ids.shape
        x_quant, x_scale, *_ = self.prepare_finalize.prepare(
            x,
            topk_weights,
            topk_ids,
            self.experts,
            None,
            False,
            self.quant_config,
            defer_input_quant=False,
        )
        assert x_scale is not None and x_scale.is_contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        ctas = max_ctas(rows, top_k, self.experts)
        permuted = torch.empty((rows, top_k), dtype=torch.int32, device=x.device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=x.device)
        _route_kernel(self.device_index)(
            *(_pointer(cutlass.Int32, t, 4) for t in (ids, permuted, record)),
            Int32(rows),
            Int32(top_k),
            Int32(self.experts),
            current_cu_stream(),
        )
        return Routing(x_quant, x_scale, ids, topk_weights, permuted, record, ctas)

    def front(
        self, x: torch.Tensor, logits: torch.Tensor, input_ids: torch.Tensor | None
    ) -> Routing:
        """Top-k, MXFP8 quantization and routing record in one launch.

        ``x``: BF16 router input [rows, 5120]; ``logits``: FP32 router logits
        [rows, 384]; ``input_ids``: [rows] int32 or int64, read only with image
        routing.
        """
        config = self.front_config
        assert config is not None
        rows = x.size(0)
        device = x.device
        ctas = max_ctas(rows, TOP_K, self.experts)
        topk_ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        topk_weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        x_quant = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=device)
        x_scale = torch.empty((rows, HIDDEN // 32), dtype=torch.uint8, device=device)
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        image = config.bias_vl is not None
        ids = topk_ids
        if image:
            assert input_ids is not None
            ids = input_ids
        _front_kernel(self.device_index, image)(
            _pointer(cutlass.Float32, logits, 16),
            _pointer(cutlass.Float32, config.bias, 16),
            _pointer(cutlass.Float32, config.bias_vl if image else config.bias, 16),
            _pointer(cutlass.Int32, ids, 4),
            Int32(int(ids.dtype == torch.int64)),
            Int32(config.image_sentinel_lo),
            Float32(config.scale),
            _pointer(cutlass.Uint32, x, 16),
            _pointer(cutlass.Int32, topk_ids, 16),
            _pointer(cutlass.Float32, topk_weights, 16),
            _pointer(cutlass.Uint8, x_quant, 16),
            _pointer(cutlass.Uint8, x_scale, 16),
            _pointer(cutlass.Int32, permuted, 16),
            _pointer(cutlass.Int32, record, 16),
            Int32(rows),
            current_cu_stream(),
        )
        return Routing(x_quant, x_scale, topk_ids, topk_weights, permuted, record, ctas)

    def gemms(self, routing: Routing) -> UnfinalizedMoEOutput:
        """Launch GEMM1 and GEMM2 right after the kernel that built ``routing``."""
        return self.gemm2(routing, *self.gemm1(routing))

    def gemm1(self, routing: Routing) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch GEMM1; returns the intermediate H and its block scales (one
        128-row scale atom per token tile, rows 0-7 used)."""
        device = routing.x_quant.device
        padded_rows = TILE * routing.ctas
        h_values = torch.empty((padded_rows, PADDED), dtype=torch.uint8, device=device)
        h_scales = torch.empty(
            routing.ctas * tile_scale_bytes(PADDED), dtype=torch.uint8, device=device
        )
        _gemm1_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w13, 32),
            _pointer(cutlass.Float8E8M0FNU, self.w13_scale, 16),
            _pointer(cutlass.Float8E4M3FN, routing.x_quant, 16),
            _pointer(cutlass.Uint32, routing.x_scale, 16),
            _pointer(cutlass.Uint16, h_values, 16),
            _pointer(cutlass.Uint8, h_scales, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            _pointer(cutlass.Float32, self.clamp, 16),
            Int32(routing.x_quant.size(0)),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(padded_rows),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return h_values, h_scales

    def gemm2(
        self, routing: Routing, h_values: torch.Tensor, h_scales: torch.Tensor
    ) -> UnfinalizedMoEOutput:
        """Launch GEMM2 on GEMM1's outputs."""
        rows, top_k = routing.topk_ids.shape
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN),
            dtype=torch.bfloat16,
            device=routing.x_quant.device,
        )
        _gemm2_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w2, 32),
            _pointer(cutlass.Float8E8M0FNU, self.w2_scale, 16),
            _pointer(cutlass.Float8E4M3FN, h_values, 16),
            _pointer(cutlass.Float8E8M0FNU, h_scales, 16),
            _pointer(cutlass.BFloat16, gemm2_rows, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            Int32(TILE * routing.ctas),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_rows,
            expert_weights=routing.topk_weights.contiguous().view(rows, top_k),
            expanded_idx_to_permuted_idx=routing.permuted,
        )


class RoutedExpertGemmNvfp4(RoutedExpertGemm):
    """The route kernel and the NVFP4 (W4A4) GEMM pair bound to one MoE
    layer's TRT-LLM-gen NVFP4 experts (``routed_expert_gemm_nvfp4_cute.py``).

    The input is quantized as the modular kernel's prepare does (NVFP4 with the
    layer's static global scale, linear E4M3 block scales), so the activations
    match the stock path bit for bit: by the fused NVFP4 front
    (``routed_expert_front_nvfp4_cute.py``: top-k, quantization and routing
    record in one launch) where the router allows it, else by the prepare and
    the route kernel on the router's top-k.
    """

    def __init__(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        clamp: torch.Tensor,
        g1_alphas: torch.Tensor,
        g1_scale_c: torch.Tensor,
        g2_alphas: torch.Tensor,
        top_k: int,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        quant_config: FusedMoEQuantConfig,
    ):
        check_weights_nvfp4(
            w13, w13_scale, w2, w2_scale, clamp, g1_alphas, g1_scale_c, g2_alphas, top_k
        )
        self.experts = int(w13.size(0))
        self.top_k = top_k
        index = w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        _route_kernel(self.device_index)
        _gemm1_nvfp4_kernel(self.device_index)
        _gemm2_nvfp4_kernel(self.device_index)
        self.prepare_finalize = prepare_finalize
        self.quant_config = quant_config
        self.front_config = None
        self.w13, self.w13_scale, self.w2, self.w2_scale = w13, w13_scale, w2, w2_scale
        self.clamp = clamp
        self.g1_alphas, self.g1_scale_c, self.g2_alphas = g1_alphas, g1_scale_c, g2_alphas

    def set_front(self, config: FrontConfig) -> None:
        """Use the fused NVFP4 front (``routed_expert_front_nvfp4_cute.py``) with
        the router's ``config``; it quantizes as the modular kernel's prepare
        does (one FP32 global scale, linear E4M3 block scales). Raises
        ``ValueError`` for another input quantization, which then keeps the
        top-k + ``route`` path."""
        # The modular prepare's scaled_fp4_quant reads element 0 of the layer's
        # input global scale (a per-expert tensor of equal values); the front
        # reads the same element.
        gscale = self.quant_config.a1_gscale
        if (
            gscale is None
            or gscale.dtype != torch.float32
            or gscale.numel() < 1
            or not gscale.is_contiguous()
        ):
            raise ValueError("the NVFP4 front needs an FP32 input global scale")
        if self.quant_config.is_scale_swizzled:
            raise ValueError("the NVFP4 front writes linear block scales only")
        _front_nvfp4_kernel(self.device_index, config.bias_vl is not None)
        self.front_config = config

    def front(
        self, x: torch.Tensor, logits: torch.Tensor, input_ids: torch.Tensor | None
    ) -> Routing:
        """Top-k, NVFP4 quantization and routing record in one launch (the
        outputs of the DSv4 top-k, ``route``'s quantization and the route
        kernel, bitwise)."""
        config = self.front_config
        assert config is not None
        rows = x.size(0)
        device = x.device
        ctas = max_ctas(rows, TOP_K, self.experts)
        topk_ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        topk_weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=device)
        x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=device)
        x_scale = torch.empty(
            (rows, HIDDEN // NVFP4_SF_VEC), dtype=torch.uint8, device=device
        )
        permuted = torch.empty((rows, TOP_K), dtype=torch.int32, device=device)
        record = torch.empty(route_words(ctas), dtype=torch.int32, device=device)
        image = config.bias_vl is not None
        ids = topk_ids
        if image:
            assert input_ids is not None
            ids = input_ids
        gscale = self.quant_config.a1_gscale
        assert gscale is not None
        _front_nvfp4_kernel(self.device_index, image)(
            _pointer(cutlass.Float32, logits, 16),
            _pointer(cutlass.Float32, config.bias, 16),
            _pointer(cutlass.Float32, config.bias_vl if image else config.bias, 16),
            _pointer(cutlass.Int32, ids, 4),
            Int32(int(ids.dtype == torch.int64)),
            Int32(config.image_sentinel_lo),
            Float32(config.scale),
            _pointer(cutlass.Uint32, x, 16),
            _pointer(cutlass.Float32, gscale, 4),
            _pointer(cutlass.Int32, topk_ids, 16),
            _pointer(cutlass.Float32, topk_weights, 16),
            _pointer(cutlass.Uint8, x_quant, 16),
            _pointer(cutlass.Uint8, x_scale, 16),
            _pointer(cutlass.Int32, permuted, 16),
            _pointer(cutlass.Int32, record, 16),
            Int32(rows),
            current_cu_stream(),
        )
        return Routing(x_quant, x_scale, topk_ids, topk_weights, permuted, record, ctas)

    def route(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Routing:
        routing = super().route(x, topk_ids, topk_weights)
        rows = x.size(0)
        if routing.x_quant.shape != (rows, HIDDEN // 2) or routing.x_scale.numel() != (
            rows * HIDDEN // NVFP4_SF_VEC
        ):
            raise ValueError(
                "unexpected NVFP4 input layout "
                f"{tuple(routing.x_quant.shape)} / {tuple(routing.x_scale.shape)}"
            )
        return routing

    def gemm1(self, routing: Routing) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch GEMM1; returns the NVFP4 intermediate H (two values a byte)
        and its block scales (one 128-row atom set per token tile)."""
        device = routing.x_quant.device
        padded_rows = TILE * routing.ctas
        h_values = torch.empty(
            (padded_rows, NV_INTERMEDIATE // 2), dtype=torch.uint8, device=device
        )
        h_scales = torch.empty(
            routing.ctas * tile_scale_bytes_nvfp4(NV_INTERMEDIATE),
            dtype=torch.uint8,
            device=device,
        )
        _gemm1_nvfp4_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w13, 32),
            _pointer(cutlass.Float8E4M3FN, self.w13_scale, 16),
            _pointer(cutlass.Float4E2M1FN, routing.x_quant, 16),
            _pointer(cutlass.Uint32, routing.x_scale, 16),
            _pointer(cutlass.Uint8, h_values, 16),
            _pointer(cutlass.Uint8, h_scales, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            _pointer(cutlass.Float32, self.clamp, 16),
            _pointer(cutlass.Float32, self.g1_alphas, 16),
            _pointer(cutlass.Float32, self.g1_scale_c, 16),
            Int32(routing.x_quant.size(0)),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(padded_rows),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return h_values, h_scales

    def gemm2(
        self, routing: Routing, h_values: torch.Tensor, h_scales: torch.Tensor
    ) -> UnfinalizedMoEOutput:
        """Launch GEMM2 on GEMM1's outputs."""
        rows, top_k = routing.topk_ids.shape
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN),
            dtype=torch.bfloat16,
            device=routing.x_quant.device,
        )
        _gemm2_nvfp4_kernel(self.device_index)(
            _pointer(cutlass.Float4E2M1FN, self.w2, 32),
            _pointer(cutlass.Float8E4M3FN, self.w2_scale, 16),
            _pointer(cutlass.Float4E2M1FN, h_values, 16),
            _pointer(cutlass.Float8E4M3FN, h_scales, 16),
            _pointer(cutlass.BFloat16, gemm2_rows, 16),
            _pointer(cutlass.Int32, routing.record, 16),
            _pointer(cutlass.Float32, self.g2_alphas, 16),
            Int32(TILE * routing.ctas),
            Int32(self.experts),
            Int32(cta_words(routing.ctas)),
            Int32(_sm_count(self.device_index)),
            current_cu_stream(),
        )
        return UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_rows,
            expert_weights=routing.topk_weights.contiguous().view(rows, top_k),
            expanded_idx_to_permuted_idx=routing.permuted,
        )


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _route_kernel(device_index: int):
    """``RoutedExpertRoute`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertRoute(),
            _fake(cutlass.Int32, 4),
            _fake(cutlass.Int32, 4),
            _fake(cutlass.Int32, 4),
            Int32(1),
            Int32(TOP_K),
            Int32(MAX_EXPERTS),
            make_fake_stream(),
        )


@cache
def _front_kernel(device_index: int, image: bool):
    """``RoutedExpertFront`` compiled for a device, with or without image
    routing."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertFront(image),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Int32, 4),
            Int32(0),
            Int32(0),
            Float32(1.0),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Int32, 16),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _front_nvfp4_kernel(device_index: int, image: bool):
    """``RoutedExpertFrontNvfp4`` compiled for a device, with or without image
    routing."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertFrontNvfp4(image),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Int32, 4),
            Int32(0),
            Int32(0),
            Float32(1.0),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Float32, 4),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Int32, 16),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm1_kernel(device_index: int):
    """``RoutedExpertGemm1`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm1(HIDDEN, SLICE, PADDED),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Uint16, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            Int32(1),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(TILE),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm2_kernel(device_index: int):
    """``RoutedExpertGemm2`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm2(HIDDEN, PADDED),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Float8E8M0FNU, 16),
            _fake(cutlass.BFloat16, 16),
            _fake(cutlass.Int32, 16),
            Int32(TILE),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm1_nvfp4_kernel(device_index: int):
    """``RoutedExpertGemm1Nvfp4`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm1Nvfp4(HIDDEN, NV_INTERMEDIATE),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Float4E2M1FN, 16),
            _fake(cutlass.Uint32, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Uint8, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            _fake(cutlass.Float32, 16),
            Int32(1),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(TILE),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm2_nvfp4_kernel(device_index: int):
    """``RoutedExpertGemm2Nvfp4`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedExpertGemm2Nvfp4(HIDDEN, NV_INTERMEDIATE),
            _fake(cutlass.Float4E2M1FN, 32),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.Float4E2M1FN, 16),
            _fake(cutlass.Float8E4M3FN, 16),
            _fake(cutlass.BFloat16, 16),
            _fake(cutlass.Int32, 16),
            _fake(cutlass.Float32, 16),
            Int32(TILE),
            Int32(MAX_EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a device: the persistent GEMMs' CTAs."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def check_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    clamp: torch.Tensor | None,
    top_k: int,
) -> None:
    """Raise ``ValueError`` unless the weights have the layout the kernels read
    and the routing fits the route kernel."""
    experts = w13.size(0)
    for tensor, shape in (
        (w13, (experts, 2 * PADDED, HIDDEN // 2)),
        (w13_scale, (experts, 2 * PADDED, HIDDEN // 32)),
        (w2, (experts, HIDDEN, PADDED // 2)),
        (w2_scale, (experts, HIDDEN, PADDED // 32)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
    if w13.data_ptr() % 32 or w2.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    if w13_scale.data_ptr() % 16 or w2_scale.data_ptr() % 16:
        raise ValueError("weight scales must be 16-byte aligned")
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(f"{experts} local experts")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top-k {top_k}")
    if (
        clamp is None
        or clamp.dtype != torch.float32
        or tuple(clamp.shape) != (experts,)
    ):
        raise ValueError("no FP32 [E] SwiGLU clamp limit")
    # GEMM1 never reads w13's padding tile (intermediate 576..639), so the
    # padding must be zero, as must the matching K columns of w2.
    if (
        torch.count_nonzero(w13.view(torch.uint8)[:, 2 * SLICE :]).item()
        or torch.count_nonzero(w2.view(torch.uint8)[:, :, SLICE // 2 :]).item()
    ):
        raise ValueError("the intermediate padding of w13/w2 is not zero")


def check_weights_nvfp4(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    clamp: torch.Tensor | None,
    g1_alphas: torch.Tensor | None,
    g1_scale_c: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    top_k: int,
) -> None:
    """Raise ``ValueError`` unless the NVFP4 weights have the layout the
    kernels read and the routing fits the route kernel."""
    experts = w13.size(0)
    i = NV_INTERMEDIATE
    for tensor, shape in (
        (w13, (experts, 2 * i, HIDDEN // 2)),
        (w13_scale, (experts, 2 * i, HIDDEN // NVFP4_SF_VEC)),
        (w2, (experts, HIDDEN, i // 2)),
        (w2_scale, (experts, HIDDEN, i // NVFP4_SF_VEC)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected a contiguous {shape} 1-byte tensor, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
    if w13.data_ptr() % 32 or w2.data_ptr() % 32:
        raise ValueError("packed expert weights must be 32-byte aligned")
    if w13_scale.data_ptr() % 16 or w2_scale.data_ptr() % 16:
        raise ValueError("weight scales must be 16-byte aligned")
    if experts % 32 or not 0 < experts <= MAX_EXPERTS:
        raise ValueError(f"{experts} local experts")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top-k {top_k}")
    for name, tensor in (
        ("SwiGLU clamp limit", clamp),
        ("g1_alphas", g1_alphas),
        ("g1_scale_c", g1_scale_c),
        ("g2_alphas", g2_alphas),
    ):
        if (
            tensor is None
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (experts,)
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            raise ValueError(f"no FP32 [E] {name}")


def _unsupported_reason_nvfp4(runner: torch.nn.Module, impl) -> str | None:
    """Why the runner's TRT-LLM-gen NVFP4 experts cannot take the NVFP4 pair."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    config = impl.moe_config
    quant = impl.quant_config
    if config.activation != MoEActivation.SILU:
        return f"activation {config.activation}"
    if impl.gemm1_alpha is not None or impl.gemm1_beta is not None:
        return "gemm1 alpha/beta"
    if impl.expects_unquantized_inputs:
        return "per-token NVFP4 activations"
    if impl.local_num_experts != config.num_experts:
        return "expert parallelism"
    if runner.routed_scaling_factor != 1.0:
        return "routed scaling"
    if quant.quant_dtype != "nvfp4" or quant.is_scale_swizzled:
        return "activations are not NVFP4 with linear scales"
    if impl.intermediate_size_per_partition != NV_INTERMEDIATE:
        return f"intermediate {impl.intermediate_size_per_partition}"
    return None


def _unsupported_reason(runner: torch.nn.Module) -> str | None:
    """Why the MoE runner's routed experts cannot take the GEMM pair."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsModular,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
    )

    kernel = runner.routed_experts.quant_method.moe_kernel
    impl = kernel.fused_experts if kernel is not None else None
    if isinstance(impl, TrtLlmNvFp4ExpertsModular):
        return _unsupported_reason_nvfp4(runner, impl)
    if not isinstance(impl, TrtLlmMxfp4ExpertsModular):
        return f"experts kernel {type(impl).__name__}"
    config = impl.moe_config
    quant = impl.quant_config
    if config.activation != MoEActivation.SILU:
        return f"activation {config.activation}"
    if impl.gemm1_alpha is not None or impl.gemm1_beta is not None:
        return "gemm1 alpha/beta"
    if impl.w1_bias is not None or impl.w2_bias is not None:
        return "expert biases"
    if impl.local_num_experts != config.num_experts:
        return "expert parallelism"
    if runner.routed_scaling_factor != 1.0:
        return "routed scaling"
    if quant.quant_dtype != "mxfp8" or quant.is_scale_swizzled:
        return "activations are not MXFP8 with linear scales"
    return None


def _front_config(runner: torch.nn.Module) -> FrontConfig | str:
    """The fused front's configuration for the runner's router, or why its
    routing is not the plain DSv4 top-k the front computes."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        FusedTopKBiasRouter,
    )

    router = runner.router
    if not isinstance(router, FusedTopKBiasRouter):
        return f"router {type(router).__name__}"
    if router.scoring_func != "sqrtsoftplus" or not router.renormalize:
        return "routing is not renormalized sqrt-softplus"
    if router.top_k != TOP_K or router.global_num_experts != EXPERTS:
        return f"top-{router.top_k} of {router.global_num_experts} experts"
    if router._hash_indices_table is not None or router.num_fused_shared_experts:
        return "hash routing or fused shared experts"
    if runner.gate is None or runner._fse_fuse_gate:
        return "no plain router gate"
    bias, bias_vl = router.e_score_correction_bias, router.bias_vl
    for tensor in (bias, bias_vl):
        if tensor is not None and (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != (EXPERTS,)
            or not tensor.is_contiguous()
        ):
            return f"bias is not FP32 [{EXPERTS}]"
    if bias is None:
        return "no score correction bias"
    if router.image_sentinel_lo <= 0:
        bias_vl = None
    return FrontConfig(
        bias=bias.data,
        bias_vl=None if bias_vl is None else bias_vl.data,
        image_sentinel_lo=0 if bias_vl is None else router.image_sentinel_lo,
        scale=float(router.routed_scaling_factor),
    )


def front_applies(gemm: RoutedExpertGemm, runner: torch.nn.Module) -> bool:
    """Whether the fused front may replace the router's top-k this call: the
    router's biases are the tensors the front reads, and no routed-expert
    capture hook (which only ``select_experts`` feeds) is set."""
    config = gemm.front_config
    router = runner.router
    if (
        config is None
        or router.capture_fn is not None
        or router._routing_replay_out is not None
    ):
        return False
    bias, bias_vl = router.e_score_correction_bias, router.bias_vl
    return (
        bias is not None
        and bias.data_ptr() == config.bias.data_ptr()
        and (
            config.bias_vl is None
            or (bias_vl is not None and bias_vl.data_ptr() == config.bias_vl.data_ptr())
        )
    )


def build_routed_expert_gemms(
    runners: list[torch.nn.Module],
) -> list[RoutedExpertGemm | None]:
    """A ``RoutedExpertGemm`` for each MoE runner that supports it (None for the
    others), with the fused front where the router allows it. Needs the MoE
    kernels that the loader's per-layer quantization finalize builds."""
    reasons: Counter[str] = Counter()
    gemms: list[RoutedExpertGemm | None] = []
    fronts = 0
    for runner in runners:
        gemm = None
        reason = _unsupported_reason(runner)
        if reason is not None:
            reasons[reason] += 1
        else:
            routed = runner.routed_experts
            kernel = routed.quant_method.moe_kernel
            impl = kernel.fused_experts
            try:
                if impl.quant_config.quant_dtype == "nvfp4":
                    quant = impl.quant_config
                    gemm = RoutedExpertGemmNvfp4(
                        routed.w13_weight,
                        quant.w1_scale,
                        routed.w2_weight,
                        quant.w2_scale,
                        impl.gemm1_clamp_limit,
                        quant.g1_alphas,
                        impl.g1_scale_c,
                        quant.g2_alphas,
                        impl.topk,
                        kernel.prepare_finalize,
                        quant,
                    )
                else:
                    gemm = RoutedExpertGemm(
                        routed.w13_weight,
                        impl.w1_scale,
                        routed.w2_weight,
                        impl.w2_scale,
                        impl.gemm1_clamp_limit,
                        impl.topk,
                        kernel.prepare_finalize,
                        impl.quant_config,
                    )
            except ValueError as error:
                reasons[str(error)] += 1
        if gemm is not None:
            front = _front_config(runner)
            if isinstance(front, str):
                reasons[f"fused front: {front}"] += 1
            else:
                try:
                    gemm.set_front(front)
                    fronts += 1
                except ValueError as error:
                    reasons[f"fused front: {error}"] += 1
        gemms.append(gemm)
    # A layer that keeps FlashInfer under decode_backend=latency_specialized is a
    # deployment problem (e.g. unexpected weights), so it warns.
    log = logger.warning if reasons else logger.info
    log(
        "DeepSeek-V4.1 low-batch routed-expert GEMMs on %d of %d MoE layers "
        "(%d with the fused front)%s.",
        sum(gemm is not None for gemm in gemms),
        len(runners),
        fronts,
        "; other layers keep FlashInfer: "
        + "; ".join(f"{reason} ({n})" for reason, n in reasons.items())
        if reasons
        else "",
    )
    return gemms
