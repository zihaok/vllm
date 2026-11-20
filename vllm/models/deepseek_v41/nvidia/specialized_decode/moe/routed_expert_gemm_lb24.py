# ruff: noqa: E501
# The 9-24-row low-batch routed-expert MoE of the TP4 decode executor for NVFP4 experts (w4a4, block 16).
#
# The executor's low-batch MoE (routed_expert_gemm.py) covers up to 8 rows; above that an NVFP4 layer runs FlashInfer
# TRT-LLM-gen (top-k, scaled_fp4_quant, routing, bmm1, bmm2: five launches). RoutedExpertGemmLB24 has RoutedExpertGemm's
# entry surface (set_front / accepts / front / route / gemms, returning UnfinalizedMoEOutput for the deferred finalize), bound
# to one layer's TrtLlmNvFp4ExpertsModular weights, for 9-24 rows (2-4 DSpark requests):
#   front:  FrontNVN2 (lb24_nvfp4_front_n.py): the executor's fused front (DSv4 top-6 + routing record) for up to 24 rows with
#           NVFP4 quantization of the routed input (scales in the 128x4 layout)
#   GEMM1:  NvGemm1NT (lb24_nvfp4_gemm_n.py): one weight pass for every row (16 / 24-column B tile), NVFP4 w4a4, clamped SwiGLU,
#           NVFP4 H
#   GEMM2:  NvGemm2V (lb24_nvfp4_gemm_v.py)
# with the per-row-count launch configuration in lb24_occupancy_table.json (data). The weights are the runner's own, as vLLM
# prepares them for FLASHINFER_TRTLLM; the scalars map as vLLM folds them: alpha1 = g1_alphas, gs = a2_gscale,
# alpha2 = g2_alphas, the clamp unfolded, x's global scale = a1_gscale (element 0, as scaled_fp4_quant reads it).
# Every GEMM2 row is bitwise equal to TRT-LLM-gen's on vLLM-prepared weights; the front's outputs are bitwise those of
# dsv4_topk + scaled_fp4_quant + the route kernel.
from __future__ import annotations

import json
import os
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_stream

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from . import lb24_kernels as LK
from . import lb24_nvfp4_front as FN
from . import lb24_nvfp4_front_n as FNN
from . import lb24_nvfp4_gemm as NV
from . import lb24_nvfp4_gemm_n as NN
from . import lb24_nvfp4_gemm_v as NVV
from . import routed_expert_gemm as REG

_HERE = os.path.dirname(os.path.abspath(__file__))

HIDDEN, SLICE, TILE, TOP_K, EXPERTS = (
    REG.HIDDEN,
    REG.SLICE,
    REG.TILE,
    REG.TOP_K,
    REG.EXPERTS,
)
MAX_ROWS_NV = 24
PADDED_NV = (
    (SLICE + 63) // 64 * 64
)  # vLLM pads the gated NVFP4 intermediate to a multiple of 64 (576 at TP4)
# rows -> (GEMM1 AB stages, GEMM1 CTAs per SM, GEMM2 AB stages, GEMM2 CTAs per SM); a call takes the smallest row >= its rows.
with open(os.path.join(_HERE, "lb24_occupancy_table.json")) as _fh:
    TABLE = {int(k): tuple(v) for k, v in json.load(_fh).items()}
F4, E4, U8, U32, I32, F32, BF = (
    cutlass.Float4E2M1FN,
    cutlass.Float8E4M3FN,
    cutlass.Uint8,
    cutlass.Uint32,
    cutlass.Int32,
    cutlass.Float32,
    cutlass.BFloat16,
)
P = REG._pointer


def _cfg(rows: int) -> tuple:
    return TABLE[min((r for r in TABLE if r >= rows), default=max(TABLE))]


def _fake(dtype, align):
    return REG._fake(dtype, align)


@cache
def _gemm1(device_index: int, cta_n: int, stages: int):
    with torch.accelerator.device_index(device_index):
        obj = (
            NV.NvGemm1(HIDDEN, SLICE, PADDED_NV, cta_k=256, num_ab_stage=stages)
            if cta_n == TILE
            else NN.NvGemm1NT(
                HIDDEN, SLICE, PADDED_NV, cta_n=cta_n, cta_k=256, num_ab_stage=stages
            )
        )
        return cute.compile(
            obj,
            _fake(F4, 32),
            _fake(E4, 16),
            _fake(F4, 32),
            _fake(E4, 16),
            _fake(U8, 16),
            _fake(U8, 16),
            _fake(I32, 16),
            _fake(F32, 16),
            _fake(F32, 16),
            _fake(F32, 16),
            Int32(1),
            Int32(REG.MAX_EXPERTS),
            Int32(4),
            Int32(TILE),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _gemm2(device_index: int, stages: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            NVV.NvGemm2V(HIDDEN, PADDED_NV, cta_k=64, num_ab_stage=stages),
            _fake(F4, 32),
            _fake(E4, 16),
            _fake(F4, 32),
            _fake(E4, 16),
            _fake(BF, 16),
            _fake(I32, 16),
            _fake(F32, 16),
            Int32(TILE),
            Int32(REG.MAX_EXPERTS),
            Int32(4),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _route_n(device_index: int):
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            LK.RoutedExpertRouteN(),
            _fake(I32, 4),
            _fake(I32, 4),
            _fake(I32, 4),
            Int32(1),
            Int32(TOP_K),
            Int32(REG.MAX_EXPERTS),
            make_fake_stream(),
        )


@cache
def _front(device_index: int, image: bool, wide: bool):
    with torch.accelerator.device_index(device_index):
        obj = FNN.FrontNVN2(image) if wide else FN.FrontNV(image)
        return cute.compile(
            obj,
            _fake(F32, 16),
            _fake(F32, 16),
            _fake(F32, 16),
            _fake(I32, 4),
            Int32(0),
            Int32(0),
            Float32(1.0),
            _fake(U32, 16),
            _fake(F32, 16),
            _fake(I32, 16),
            _fake(F32, 16),
            _fake(U8, 16),
            _fake(U8, 16),
            _fake(I32, 16),
            _fake(I32, 16),
            Int32(1),
            make_fake_stream(),
        )


def check_weights_nv(w13, w13_scale, w2, w2_scale, scalars, top_k) -> None:
    """The NVFP4 weights the build reads, as vLLM prepares them for FLASHINFER_TRTLLM at the TP4 rank's width."""
    e = w13.size(0)
    want = {
        "w13": (w13, (e, 2 * PADDED_NV, HIDDEN // 2)),
        "w13_scale": (w13_scale, (e, 2 * PADDED_NV, HIDDEN // 16)),
        "w2": (w2, (e, HIDDEN, PADDED_NV // 2)),
        "w2_scale": (w2_scale, (e, HIDDEN, PADDED_NV // 16)),
    }
    for name, (t, shape) in want.items():
        if tuple(t.shape) != shape or not t.is_contiguous() or t.element_size() != 1:
            raise ValueError(f"nvfp4 {name} shape {tuple(t.shape)} != {shape}")
    for name, t in scalars.items():
        if t is None or t.numel() not in (1, e) or t.dtype != torch.float32:
            raise ValueError(f"nvfp4 scalar {name}")
    if top_k != TOP_K or e % 32 or e > REG.MAX_EXPERTS:
        raise ValueError("nvfp4 top_k / experts")


def _per_expert(t: torch.Tensor, e: int) -> torch.Tensor:
    return (t.expand(e) if t.numel() == 1 else t).float().contiguous()


class RoutedExpertGemmLB24:
    """The 9-24-row NVFP4 low-batch MoE bound to one layer's TRT-LLM-gen NVFP4 experts (module header)."""

    max_rows = MAX_ROWS_NV

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        clamp,
        g1_alphas,
        a2_gscale,
        g2_alphas,
        a1_gscale,
        top_k: int,
    ) -> None:
        e = int(w13.size(0))
        check_weights_nv(
            w13,
            w13_scale,
            w2,
            w2_scale,
            {
                "clamp": clamp,
                "g1_alphas": g1_alphas,
                "a2_gscale": a2_gscale,
                "g2_alphas": g2_alphas,
                "a1_gscale": a1_gscale,
            },
            top_k,
        )
        self.experts, self.top_k = e, top_k
        index = w13.device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        self.w13, self.w2 = w13, w2
        self.w13_scale = w13_scale.view(torch.uint8).reshape(-1)
        self.w2_scale = w2_scale.view(torch.uint8).reshape(-1)
        self.clamp = _per_expert(clamp, e)
        self.alpha1 = _per_expert(g1_alphas, e)
        self.gs = _per_expert(a2_gscale, e)
        self.alpha2 = _per_expert(g2_alphas, e)
        self.x_gs = a1_gscale.reshape(-1)[:1].float().contiguous()
        self.front_config: REG.FrontConfig | None = None
        self.sm = REG._sm_count(self.device_index)
        # Compile every kernel now, not inside a CUDA graph capture.
        for rows, (s1, _m1, s2, _m2) in TABLE.items():
            for cta_n in (TILE,) if rows <= TILE else (16, 24):
                _gemm1(self.device_index, cta_n, s1)
            _gemm2(self.device_index, s2)
        REG._route_kernel(self.device_index)
        _route_n(self.device_index)

    def set_front(self, config: REG.FrontConfig) -> None:
        image = config.bias_vl is not None
        _front(self.device_index, image, False)
        _front(self.device_index, image, True)
        self.front_config = config

    def accepts(self, x: torch.Tensor, runner: torch.nn.Module) -> bool:
        """1-24 BF16 rows, and the runner's expert weights still the tensors bound here."""
        return (
            1 <= x.size(0) <= MAX_ROWS_NV
            and x.dtype == torch.bfloat16
            and tuple(x.shape) == (x.size(0), HIDDEN)
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and runner.routed_experts.w13_weight.data_ptr() == self.w13.data_ptr()
        )

    def _buffers(self, rows: int):
        dev = self.w13.device
        ctas = REG.max_ctas(rows, self.top_k, self.experts)
        return (
            ctas,
            torch.empty((rows, self.top_k), dtype=torch.int32, device=dev),
            torch.empty(REG.route_words(ctas), dtype=torch.int32, device=dev),
        )

    def route(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> REG.Routing:
        """NVFP4-quantize ``x`` (scaled_fp4_quant with the layer's input global scale, scales in the 128x4 layout) and route."""
        from vllm import _custom_ops as ops

        rows, top_k = topk_ids.shape
        x_quant, x_scale = ops.scaled_fp4_quant(
            x, self.x_gs, is_sf_swizzled_layout=True
        )
        ids = topk_ids.to(torch.int32).contiguous()
        ctas, permuted, record = self._buffers(rows)
        kernel = (
            REG._route_kernel(self.device_index)
            if rows <= TILE
            else _route_n(self.device_index)
        )
        kernel(
            P(I32, ids, 4),
            P(I32, permuted, 4),
            P(I32, record, 4),
            Int32(rows),
            Int32(top_k),
            Int32(self.experts),
            current_cu_stream(),
        )
        return REG.Routing(x_quant, x_scale, ids, topk_weights, permuted, record, ctas)

    def front(
        self, x: torch.Tensor, logits: torch.Tensor, input_ids: torch.Tensor | None
    ) -> REG.Routing:
        """Top-k, NVFP4 quantization and routing record in one launch."""
        config = self.front_config
        assert config is not None
        rows = x.size(0)
        dev = x.device
        ctas, permuted, record = self._buffers(rows)
        topk_ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=dev)
        topk_weights = torch.empty((rows, TOP_K), dtype=torch.float32, device=dev)
        x_quant = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8, device=dev)
        x_scale = torch.full(
            (FN.SCALE_BYTES,), NV.E4M3_ONE, dtype=torch.uint8, device=dev
        )
        image = config.bias_vl is not None
        ids = topk_ids
        if image:
            assert input_ids is not None
            ids = input_ids
        _front(self.device_index, image, rows > TILE)(
            P(F32, logits, 16),
            P(F32, config.bias, 16),
            P(F32, config.bias_vl if image else config.bias, 16),
            P(I32, ids, 4),
            Int32(int(ids.dtype == torch.int64)),
            Int32(config.image_sentinel_lo),
            Float32(config.scale),
            P(U32, x, 16),
            P(F32, self.x_gs, 16),
            P(I32, topk_ids, 16),
            P(F32, topk_weights, 16),
            P(U8, x_quant, 16),
            P(U8, x_scale, 16),
            P(I32, permuted, 16),
            P(I32, record, 16),
            Int32(rows),
            current_cu_stream(),
        )
        return REG.Routing(
            x_quant, x_scale, topk_ids, topk_weights, permuted, record, ctas
        )

    def gemms(self, routing: REG.Routing) -> UnfinalizedMoEOutput:
        """GEMM1 then GEMM2 right after the kernel that built ``routing``, launched per the table."""
        rows, top_k = routing.topk_ids.shape
        s1, m1, s2, m2 = _cfg(rows)
        cta_n = TILE if rows <= TILE else (16 if rows <= 16 else 24)
        dev = routing.x_quant.device
        hr = TILE * routing.ctas
        cw = REG.cta_words(routing.ctas)
        h = torch.empty((hr, PADDED_NV // 2), dtype=torch.uint8, device=dev)
        hs = torch.empty(
            routing.ctas * NV.nv_tile_scale_bytes(PADDED_NV),
            dtype=torch.uint8,
            device=dev,
        )
        _gemm1(self.device_index, cta_n, s1)(
            P(F4, self.w13, 32),
            P(E4, self.w13_scale, 16),
            P(F4, routing.x_quant, 32),
            P(E4, routing.x_scale, 16),
            P(U8, h, 16),
            P(U8, hs, 16),
            P(I32, routing.record, 16),
            P(F32, self.clamp, 16),
            P(F32, self.alpha1, 16),
            P(F32, self.gs, 16),
            Int32(rows),
            Int32(self.experts),
            Int32(cw),
            Int32(hr),
            Int32(self.sm * m1),
            current_cu_stream(),
        )
        gemm2_rows = torch.empty(
            (rows * top_k * TILE, HIDDEN), dtype=torch.bfloat16, device=dev
        )
        _gemm2(self.device_index, s2)(
            P(F4, self.w2, 32),
            P(E4, self.w2_scale, 16),
            P(F4, h, 32),
            P(E4, hs, 16),
            P(BF, gemm2_rows, 16),
            P(I32, routing.record, 16),
            P(F32, self.alpha2, 16),
            Int32(hr),
            Int32(self.experts),
            Int32(cw),
            Int32(self.sm * m2),
            current_cu_stream(),
        )
        return UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_rows,
            expert_weights=routing.topk_weights.contiguous().view(rows, top_k),
            expanded_idx_to_permuted_idx=routing.permuted,
        )
