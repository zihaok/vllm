# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed experts with a fused MXFP8 shared expert on NVIDIA's Rubin MegaMoE kernel.

``VllmNvfp4MegaMoEExperts`` replaces ``DeepGemmMegaMoEExperts``. A layer is NVFP4
when the model's quantization config says its routed experts are NVFP4
(``_is_nvfp4_expert_layer``, e.g. ``moe_quant_algo: NVFP4`` in
nvidia/DeepSeek-V4.1-Flash-NVFP4) and it is a target layer (index below
``num_hidden_layers``).

NVFP4 layers:

* register the eight NVFP4 checkpoint tensors the DeepSeek-V4.1 loader indexes
  (``w13/w2`` ``weight``, ``weight_scale``, ``weight_scale_2``, ``input_scale``)
  and write every shard straight into the kernel layout (``layout.py``); the
  shared expert is stored as MXFP8 with UE8M0 1x32 scales in the same layouts;
* quantize tokens with the layer's static input global scale ``1 / A1``
  (``A1`` = max ``w1``/``w3`` ``input_scale`` over all experts, the same on every
  rank), and fold it into the FC1 alpha: ``fc1_alpha = A1 * weight_scale_2``;
* requantize the FC1 output (top-k weight applied) with ``1 / A2`` (``A2`` = max
  ``w2`` ``input_scale``), folded into ``fc2_alpha = A2 * w2 weight_scale_2``;
* run one staging launch plus the kernel (router, fused routed + shared FFN,
  top-k reduce adding the shared-expert output) per forward.

EPLB (``--enable-eplb``) on NVFP4 layers:

* the router hands the layer LOGICAL expert ids; FlashInfer staging maps them
  to physical replicas and records the expert load with vLLM's own
  ``eplb_map_and_record`` (native vLLM does this in its own staging);
* ``fc1_alpha`` / ``fc2_alpha`` are per-slot buffers allocated at construction and
  returned by ``get_expert_weights`` next to the eight checkpoint tensors, so EPLB
  moves them in place with their experts (it has no post-move hook);
* the kernel needs no change: it addresses experts by physical id over contiguous
  per-rank blocks and its tuned settings are looked up by the logical count.
"""

import atexit
import collections
import logging
import math
import os
import re
import types
import weakref
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_NVFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)

from . import layout
from .compatibility import ensure_quack_compatibility
from flashinfer.moe_ep.sm107_nvfp4 import (
    bucket_for,
    decode_kernel,
    geometry_unsupported_reason,
    l2_margin,
    layer_buckets,
    scale_multiplier,
)
from .eplb import PLUGIN as _NVFP4_PLUGIN
from .eplb import UNSUPPORTED_EPLB_COMMUNICATORS as _NVFP4_UNSUPPORTED_EPLB  # noqa: F401 (tests classify with it)
from .eplb import check_eplb_args, check_eplb_policy, eplb_kernel_args, view_signature

logger = logging.getLogger("vllm.flashinfer.sm107_nvfp4")

_TARGET_EXPERTS_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts$")
_ROUTED_NAMES = (
    "w13_weight", "w13_weight_scale", "w13_weight_scale_2", "w13_input_scale",
    "w2_weight", "w2_weight_scale", "w2_weight_scale_2", "w2_input_scale",
)
# Per-slot fp32 buffers derived from weight_scale_2 that EPLB must move with the experts (appended to the views).
_EPLB_ALPHA_NAMES = ("_fc1_alpha", "_fc2_alpha")
_eplb_logged: set[tuple] = set()

# FRONT2 (opt-in, default off): above the GateLinear threshold, where the model's gate is DeepGEMM's bf16_mega_gate,
# one DeepGEMM bf16_mega_gate_front2 launch runs that gate (same config, top-k and weights; bias, image bias and mask),
# the EPLB map and load record and this runtime's staging, writing the planes run() would stage (staging rule 1 =
# stage_nvfp4_mxfp8's bytes); the kernel then runs on them (runtime.launch). Hash-routed layers keep the unfused front.
FRONT2_ENV = "VLLM_NVFP4_MEGAMOE_FRONT2"
FRONT2_STAGING_RULE = 1  # bf16_mega_gate_front2 staging_rule: the SM107 runtime's Triton staging
_front2_census: dict[str, Any] = {"layers": collections.Counter(), "calls": 0, "rank": "?", "atexit": False}


def front2_enabled() -> bool:
    value = os.environ.get(FRONT2_ENV, "0").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{FRONT2_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _front2_census_line(tag: str) -> None:
    layers = ",".join(f"{k}:{v}" for k, v in sorted(_front2_census["layers"].items())) or "none"
    logger.info("NVFP4 MegaMoE FRONT2 census tag=%s rank=%s layers=%s calls=%d", tag, _front2_census["rank"], layers,
                _front2_census["calls"])


def is_nvfp4_routed_layer(vllm_config: Any, prefix: str) -> bool:
    """Whether the experts at ``prefix`` are NVFP4 target-model experts."""
    quant_config = getattr(vllm_config, "quant_config", None)
    predicate = getattr(quant_config, "_is_nvfp4_expert_layer", None)
    if predicate is None or not predicate(prefix):
        return False
    match = _TARGET_EXPERTS_RE.search(prefix)
    if match is None:
        return False
    num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    return int(match.group(1)) < num_layers


def check_uniform_target_formats(vllm_config: Any, prefix: str) -> None:
    """With EPLB, refuse a target model whose MoE layers are not all NVFP4 or all MXFP4.

    EPLB sizes its receive buffer from layer 0's ``get_expert_weights()`` and zips every
    layer's list with it, which silently truncates: an NVFP4 layer exposes 10 views
    (8 checkpoint tensors + 2 alphas), an MXFP4 layer exposes a different set of views.
    Checked at construction of every target layer, before any allocation. Draft layers
    (index >= num_hidden_layers) are a separate EPLB model or not registered at all.
    """
    quant_config = getattr(vllm_config, "quant_config", None)
    predicate = getattr(quant_config, "_is_nvfp4_expert_layer", None)
    match = _TARGET_EXPERTS_RE.search(prefix)
    if predicate is None or match is None:
        return
    num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    if int(match.group(1)) >= num_layers:
        return
    head, tail = prefix[: match.start(1)], prefix[match.end(1):]
    nvfp4 = [bool(predicate(f"{head}{index}{tail}")) for index in range(num_layers)]
    if any(nvfp4) and not all(nvfp4):
        mxfp4 = [index for index, is_nvfp4 in enumerate(nvfp4) if not is_nvfp4]
        raise NotImplementedError(
            f"{_NVFP4_PLUGIN}: EPLB needs every target MoE layer in one expert format, but layers {mxfp4} are not "
            f"NVFP4 (quantization ignore list) while the others are; mixed NVFP4/MXFP4 checkpoints are unsupported "
            f"with --enable-eplb"
        )


def eplb_model_key(vllm_config: Any, prefix: str) -> str:
    """Which EPLB model a layer belongs to: the module path before ``layers.N``, target or draft."""
    match = _TARGET_EXPERTS_RE.search(prefix)
    if match is None:
        return prefix
    root = prefix[: match.start()]
    text_config = getattr(getattr(vllm_config, "model_config", None), "hf_text_config", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers is None:
        return root
    return f"{root}:{'target' if int(match.group(1)) < num_layers else 'draft'}"


class VllmNvfp4MegaMoEExperts(DeepGemmMegaMoEExperts):
    """NVFP4 routed experts and a fused MXFP8 shared expert via FlashInfer."""

    _synchronized_ep_groups: ClassVar[set[tuple[int, int]]] = set()
    _eplb_enabled: bool = False
    _eplb_model_key: str = ""
    _nvfp4_runtimes: ClassVar[dict[tuple, Any]] = {}
    # EPLB view signature per model (eplb_model_key + engine):
    # (weak reference to the first layer, its prefix, signature). An entry whose first layer is gone is stale.
    _eplb_view_signatures: ClassVar[dict[str, tuple[Any, str, tuple]]] = {}

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any):
        parallel_config = vllm_config.parallel_config
        prefix = kwargs.get("prefix", "")
        # Scoped to this engine: static_forward_context is one dict per VllmConfig, shared by all its layers, so a
        # second model built in the same process (another EP size or redundancy) does not inherit this signature.
        context = getattr(getattr(vllm_config, "compilation_config", None), "static_forward_context", None)
        model_key = eplb_model_key(vllm_config, prefix)
        if context is not None:
            model_key = f"{model_key}@{id(context):x}"
        object.__setattr__(self, "_eplb_model_key", model_key)
        if parallel_config.enable_eplb:
            check_uniform_target_formats(vllm_config, prefix)
        if not is_nvfp4_routed_layer(vllm_config, prefix):
            raise ValueError(f"{prefix}: expected NVFP4 target experts")
        if parallel_config.enable_eplb or getattr(parallel_config, "enable_elastic_ep", False):
            check_eplb_policy(parallel_config)
        activation = kwargs["activation"]
        if activation.name != "swiglu" or activation.alpha != 1.0 or activation.beta != 0.0:
            raise NotImplementedError(f"FlashInfer NVFP4 MegaMoE supports plain or clamped SwiGLU, not {activation}")
        kwargs["weight_format"] = MEGA_MOE_NVFP4
        super().__init__(vllm_config, **kwargs)
        if not self.has_fused_shared_experts:
            raise NotImplementedError(
                f"{self.prefix}: FlashInfer NVFP4 MegaMoE needs the fused MXFP8 shared expert "
                "(unsharded shared MLP, VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION unset)"
            )
        # input_scale of every logical expert (all ranks see all of them): the kernel takes one global
        # FC1-input scale per layer (tokens of any rank reach any expert) and one FC2-input scale per layer.
        self._a1_in = torch.zeros(self.num_logical_experts, 2, dtype=torch.float32)
        self._a2_in = torch.zeros(self.num_logical_experts, dtype=torch.float32)
        self._ws2_13 = torch.zeros(self.num_logical_experts, 2, dtype=torch.float32)
        self._nvfp4_weights = None
        self._nvfp4_kwargs = None
        self._nvfp4_buckets = layer_buckets(self.max_num_tokens)
        self._eplb_enabled = bool(parallel_config.enable_eplb)
        logger.info(
            "%s: NVFP4 MegaMoE (NVIDIA Rubin kernel, W4A4 routed + fused MXFP8 shared expert), buckets %s",
            self.prefix, self._nvfp4_buckets,
        )
        if self._eplb_enabled:
            key = (self.num_experts, self.num_logical_experts, self.num_local_experts)
            if key not in _eplb_logged:
                _eplb_logged.add(key)
                logger.info(
                    "NVFP4 MegaMoE EPLB: %d physical experts (%d per rank), %d logical, communicator %s",
                    self.num_experts, self.num_local_experts, self.num_logical_experts,
                    parallel_config.eplb_config.communicator,
                )

    # ---------------------------------------------------------------------------------------------- weights
    def _kernel_fuses_shared_experts(self, spec: MegaMoEWeightSpec) -> bool:
        return spec.dtype == torch.float8_e4m3fn and spec.scale_group_k == 32

    def _create_weights(self) -> None:
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        # Before any allocation: EPLB layouts (redundant experts) only at (physical experts, EP) pairs that ran on
        # GPUs (config.NVFP4_GEOMETRIES); layouts without redundant experts are admitted as before.
        reason = geometry_unsupported_reason(self.num_experts, get_ep_group().world_size, h, i, self.top_k,
                                             logical_experts=self.num_logical_experts)
        if reason is not None:
            raise NotImplementedError(f"{self.prefix}: {reason}")

        def register(name: str, data: torch.Tensor) -> None:
            self.register_parameter(name, nn.Parameter(data, requires_grad=False))

        # Kernel storage: FC1 rows in 16-row gate/up stripes, scales flat 32x4x4-swizzled.
        register("w13_weight", torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8))
        register("w13_weight_scale", torch.zeros(e, 2 * i * (h // 16), dtype=torch.uint8))
        register("w13_weight_scale_2", torch.zeros(e, 2, dtype=torch.float32))
        register("w13_input_scale", torch.zeros(e, 2, dtype=torch.float32))
        register("w2_weight", torch.zeros(e, h, i // 2, dtype=torch.uint8))
        register("w2_weight_scale", torch.zeros(e, h * (i // 16), dtype=torch.uint8))
        register("w2_weight_scale_2", torch.zeros(e, dtype=torch.float32))
        register("w2_input_scale", torch.zeros(e, dtype=torch.float32))
        if self.shared_spec is not None:
            s = i * self.num_shared_experts
            register("shared_l1_weight", torch.zeros(2 * s, h, dtype=torch.uint8))
            register("shared_l1_weight_scale", torch.zeros(2 * s * (h // 32), dtype=torch.uint8))
            register("shared_l2_weight", torch.zeros(h, s, dtype=torch.uint8))
            register("shared_l2_weight_scale", torch.zeros(h * (s // 32), dtype=torch.uint8))
        # Per-slot kernel scalars, allocated once here and filled in place by _finalize_nvfp4: stable addresses for
        # CUDA graphs and for EPLB, which collects get_expert_weights() (alphas included) in load_model, before the
        # lazy finalize, and moves the rows with their experts. Not checkpoint tensors (non-persistent buffers).
        for name in (*_EPLB_ALPHA_NAMES, "_fc1_norm_const"):
            self.register_buffer(name, torch.zeros(e, dtype=torch.float32), persistent=False)

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Local slots of logical expert ``expert_id``, for every weight loader.

        Before EPLB is set up this is vLLM's initial placement (physical ``p``
        holds logical ``p % num_logical_experts``). Afterwards EPLB may have
        moved the expert, so the current replica map decides; reloads are the
        only callers then, and each pays one small device-to-host copy.
        """
        mapping = self.eplb_state.logical_to_physical_map
        if mapping is None:
            return super()._map_global_expert_id(expert_id)
        return [
            p - self.experts_start_idx
            for p in mapping[expert_id].tolist()
            if self.experts_start_idx <= p < self.experts_end_idx
        ]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        is_global = weight_name.endswith(("weight_scale_2", "input_scale"))
        if is_global and not loaded_weight.is_meta:
            value = float(loaded_weight.float().reshape(-1)[0])
            if weight_name.endswith("input_scale"):
                if shard_id == "w2":
                    self._a2_in[expert_id] = value
                else:
                    self._a1_in[expert_id, layout.gate_up_half(shard_id)] = value
            elif shard_id != "w2":
                self._ws2_13[expert_id, layout.gate_up_half(shard_id)] = value
            self._nvfp4_kwargs = None
        local_ids = self._map_global_expert_id(expert_id)
        prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_ids or prefix not in weight_name:
            return False if return_success else None
        h, i = self.hidden_size, self.intermediate_size
        for local_id in local_ids:
            dst = param.data[local_id]
            if is_global:
                value = loaded_weight.reshape(()).to(torch.float32)
                (dst if shard_id == "w2" else dst[layout.gate_up_half(shard_id)]).copy_(value)
            elif weight_name.endswith("weight_scale"):
                scale = loaded_weight.view(torch.uint8)
                if shard_id == "w2":
                    if tuple(scale.shape) != (h, i // 16):
                        raise ValueError(f"w2 weight_scale {tuple(scale.shape)} != {(h, i // 16)}")
                    layout.write_plain(dst, scale)
                else:
                    if tuple(scale.shape) != (i, h // 16):
                        raise ValueError(f"{shard_id} weight_scale {tuple(scale.shape)} != {(i, h // 16)}")
                    layout.write_gate_up_half(dst, scale, layout.gate_up_half(shard_id))
            elif weight_name.endswith("weight"):
                data = loaded_weight.view(torch.uint8)
                if shard_id == "w2":
                    if tuple(data.shape) != (h, i // 2):
                        raise ValueError(f"w2 weight {tuple(data.shape)} != {(h, i // 2)}")
                    dst.copy_(data)
                else:
                    if tuple(data.shape) != (i, h // 2):
                        raise ValueError(f"{shard_id} weight {tuple(data.shape)} != {(i, h // 2)}")
                    layout.write_gate_up_data(dst, data, layout.gate_up_half(shard_id))
            else:
                raise ValueError(f"Unexpected NVFP4 expert tensor {weight_name}")
        self._nvfp4_kwargs = None
        return True if return_success else None

    def shared_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str) -> None:
        shard_id = "w1" if ".w1." in weight_name else "w3" if ".w3." in weight_name else "w2"
        h, s = self.hidden_size, self.intermediate_size * self.num_shared_experts
        rows, cols = (h, s) if shard_id == "w2" else (s, h)
        if "weight_scale" in weight_name:
            scale = layout.expand_block_scales(loaded_weight, rows, cols, 32)
            if shard_id == "w2":
                layout.write_plain(param.data, scale)
            else:
                layout.write_gate_up_half(param.data, scale, layout.gate_up_half(shard_id))
        else:
            data = loaded_weight.view(torch.uint8)
            if tuple(data.shape) != (rows, cols):
                raise ValueError(f"shared {shard_id} weight {tuple(data.shape)} != {(rows, cols)}")
            if shard_id == "w2":
                param.data.copy_(data)
            else:
                layout.write_gate_up_data(param.data, data, layout.gate_up_half(shard_id))
        self._nvfp4_kwargs = None

    def _fill_scale_buffers(self) -> tuple[float, float]:
        """Check the loaded global scales and write the per-slot alphas in place; returns ``(A1, A2)``.

        ``A1``/``A2`` are maxima over all logical experts, so they do not depend on the placement; the alphas
        are computed from the ``weight_scale_2`` rows currently in each slot. Same arithmetic as before EPLB
        support (bit-identical kernel scalars).
        """
        if not torch.equal(self._ws2_13[:, 0], self._ws2_13[:, 1]):
            worst = int((self._ws2_13[:, 0] - self._ws2_13[:, 1]).abs().argmax())
            raise ValueError(
                f"{self.prefix}: expert {worst} has w1 weight_scale_2 {float(self._ws2_13[worst, 0])} != w3 "
                f"{float(self._ws2_13[worst, 1])}; the kernel takes one FC1 scale per expert"
            )
        a1 = float(self._a1_in.max())
        a2 = float(self._a2_in.max()) * l2_margin()
        if not (a1 > 0 and a2 > 0 and math.isfinite(a1) and math.isfinite(a2)):
            raise ValueError(f"{self.prefix}: NVFP4 input scales not loaded (A1={a1}, A2={a2})")
        if self._eplb_enabled and not self.__dict__.get("_global_scales_checked", False):
            # A collective: only at the first finalize, which every rank runs in lockstep (a later lazy re-finalize
            # after a partial reload might run on some ranks only).
            self._check_global_scales_agree(a1, a2)
            self._global_scales_checked = True
        self._place_scale_buffers(self.w13_weight.device)
        ws2_13 = self.w13_weight_scale_2.data[:, 0].to(torch.float32)
        self._fc1_alpha.copy_(ws2_13 * a1)
        self._fc2_alpha.copy_(self.w2_weight_scale_2.data.to(torch.float32) * a2)
        self._fc1_norm_const.fill_(1.0 / a2)
        return a1, a2

    def _place_scale_buffers(self, device: torch.device) -> None:
        """Keep the alpha buffers on the weights' device (they are allocated where the module is built)."""
        if self._fc1_alpha.device == device:
            return
        if self._eplb_enabled and self.eplb_state.logical_to_physical_map is not None:
            raise RuntimeError(
                f"{self.prefix}: the NVFP4 alpha buffers are on {self._fc1_alpha.device}, the weights on {device}, "
                "after EPLB collected them; build the layer on the weights' device"
            )
        for name in (*_EPLB_ALPHA_NAMES, "_fc1_norm_const"):  # e.g. built on meta; filled right after
            setattr(self, name, torch.zeros(self.num_local_experts, dtype=torch.float32, device=device))

    def _check_global_scales_agree(self, a1: float, a2: float) -> None:
        """With EPLB, alpha rows computed on one rank are copied to others: A1/A2 must agree on every rank.

        Each rank records the input_scale of every logical expert, so they agree unless a loader skipped some
        (e.g. meta tensors); then max() would silently differ between ranks.
        """
        ep_group = get_ep_group()
        if ep_group.world_size == 1:
            return
        import torch.distributed as dist

        values = torch.tensor([a1, a2], dtype=torch.float64)
        low, high = values.clone(), values.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=ep_group.cpu_group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=ep_group.cpu_group)
        if not torch.equal(low, high):
            raise RuntimeError(
                f"{self.prefix}: NVFP4 global input scales differ across EP ranks (A1 in [{low[0]}, {high[0]}], "
                f"A2 in [{low[1]}, {high[1]}]); EPLB moves alpha rows between ranks, so every rank must load the "
                "input_scale of every expert"
            )

    def _finalize_nvfp4(self) -> None:
        """Global scales and kernel views after loading (eager; before CUDA graph capture).

        The alpha buffers are written in place, so kernels and graphs bound to them stay valid across a re-finalize
        (reload). The re-finalize itself only runs from an eager ``forward`` (graph replays skip this Python): after a
        reload under CUDA graphs, run one eager forward on every rank (or re-capture) before serving, or replays keep
        the old alphas next to the new weight_scale_2. ``1 / A1`` is passed to staging by value and baked into
        captured graphs: a reload that changes a w1/w3 input_scale under CUDA graphs needs re-capture.
        """
        if self._nvfp4_kwargs is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{self.prefix}: NVFP4 MegaMoE weights must be finalized before graph capture")
        a1, a2 = self._fill_scale_buffers()
        weights = types.SimpleNamespace(
            w13=self.w13_weight.data, w13_sf=self.w13_weight_scale.data,
            w2=self.w2_weight.data, w2_sf=self.w2_weight_scale.data,
            fc1_alpha=self._fc1_alpha, fc2_alpha=self._fc2_alpha, fc1_norm_const=self._fc1_norm_const,
            shared_w13=self.shared_l1_weight.data, shared_w13_sf=self.shared_l1_weight_scale.data,
            shared_w2=self.shared_l2_weight.data, shared_w2_sf=self.shared_l2_weight_scale.data,
        )
        self._nvfp4_weights = weights
        self._input_norm_const = 1.0 / a1
        self._scale_mul = scale_multiplier(self._input_norm_const)
        from flashinfer.moe_ep.sm107_nvfp4 import Nvfp4MegaMoERuntime

        self._nvfp4_kwargs = Nvfp4MegaMoERuntime.layer_kwargs(weights)
        logger.debug("%s: NVFP4 MegaMoE A1=%.4e A2=%.4e", self.prefix, a1, a2)

    # ------------------------------------------------------------------------------------------------ runtime
    def get_symm_buffer(self):
        runtime = self.__dict__.get("_nvfp4_runtime")
        if runtime is not None and self._nvfp4_kwargs is not None:
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up FlashInfer NVFP4 MegaMoE before CUDA graph capture")
        self._finalize_nvfp4()
        ep_group = get_ep_group()
        key = (
            id(ep_group.device_group), torch.cuda.current_device(), self.num_experts, self.num_logical_experts,
            self.max_num_tokens, self.top_k, self.hidden_size, self.intermediate_size, self.num_shared_experts,
            self.activation.clamp,
        )
        runtime = self._nvfp4_runtimes.get(key)
        if runtime is None:
            ensure_quack_compatibility()
            if decode_kernel(self.max_num_tokens) == "genphase":
                from flashinfer.moe_ep.sm107_nvfp4 import GenphaseRuntime as Runtime
            else:
                from flashinfer.moe_ep.sm107_nvfp4 import Nvfp4MegaMoERuntime as Runtime

            runtime = Runtime(
                ep_group=ep_group, num_experts=self.num_experts, hidden=self.hidden_size,
                intermediate=self.intermediate_size,
                shared_intermediate=self.intermediate_size * self.num_shared_experts, topk=self.top_k,
                clamp=self.activation.clamp, max_num_tokens=self.max_num_tokens,
                num_logical_experts=self.num_logical_experts,
            )
            runtime.compile_all(self._nvfp4_kwargs)
            self._nvfp4_runtimes[key] = runtime
        # The kernel addresses experts by physical id over contiguous per-rank blocks: it must be built for this
        # layer's physical count and slots (staging hands it physical ids when EPLB is on).
        if (runtime.num_experts, runtime.num_local_experts) != (self.num_experts, self.num_local_experts):
            raise RuntimeError(
                f"{self.prefix}: NVFP4 MegaMoE runtime built for {runtime.num_experts} experts "
                f"({runtime.num_local_experts} per rank), layer has {self.num_experts} ({self.num_local_experts})"
            )
        object.__setattr__(self, "_nvfp4_runtime", runtime)
        return runtime

    def get_block_m(self, num_tokens: int) -> int:
        return bucket_for(self._nvfp4_buckets, num_tokens)

    def get_expert_weights(self) -> list[torch.Tensor]:
        """Routed tensors as ``(num_local_experts, -1)`` views for EPLB, which moves them row by row in place.

        NVFP4 layers: the eight checkpoint tensors (vLLM's native views, shared expert excluded), then the
        ``fc1_alpha`` and ``fc2_alpha`` buffers, so each slot's kernel scalars follow its expert. EPLB has no
        post-move hook; a moved expert with its old slot's alphas would run with ws2[old] / ws2[new] scaling.
        All layers must expose the same view signature because EPLB shapes its buffers from the first layer.
        """
        e = self.num_local_experts
        names = tuple(n for n, _ in self.named_parameters(recurse=False) if not n.startswith("shared_"))
        if names != _ROUTED_NAMES:
            raise AssertionError(f"{self.prefix}: NVFP4 routed parameters {names}, expected {_ROUTED_NAMES}")
        views = super().get_expert_weights()
        views += [getattr(self, name).view(e, 1) for name in _EPLB_ALPHA_NAMES]
        sources = [getattr(self, name).data for name in names] + [getattr(self, n) for n in _EPLB_ALPHA_NAMES]
        for name, view, tensor in zip(names + _EPLB_ALPHA_NAMES, views, sources, strict=True):
            # EPLB writes the kernel's memory through the view: it must cover the tensor exactly, in place.
            if (
                view.data_ptr() != tensor.data_ptr()
                or view.numel() != tensor.numel()
                or not view.is_contiguous()
                or view.dim() != 2
                or view.shape[0] != e
            ):
                raise AssertionError(
                    f"{self.prefix}: NVFP4 MegaMoE EPLB {name}: the view does not cover its memory in place "
                    f"(shape {tuple(view.shape)}, tensor {tuple(tensor.shape)})"
                )
        self._check_eplb_signature(views)
        return views

    def _check_eplb_signature(self, views: list[torch.Tensor]) -> None:
        signature = view_signature(views)
        entry = self._eplb_view_signatures.get(self._eplb_model_key)
        if entry is None or entry[0]() is None:  # first layer of this model, or a dead model's entry (id reuse)
            self._eplb_view_signatures[self._eplb_model_key] = (weakref.ref(self), self.prefix, signature)
            return
        _, first, known = entry
        if known != signature:
            raise RuntimeError(
                f"{self.prefix}: EPLB views {signature} differ from those of {first} {known}; EPLB sizes its "
                "buffers from the first layer and would silently truncate or misplace the rest (mixed NVFP4/MXFP4 "
                "layers are unsupported with EPLB)"
            )

    def _bucket_tokens(self, num_tokens: int) -> int:
        """Tokens that size the kernel variant: the most any DP rank runs, so every EP rank agrees."""
        if is_forward_context_available():
            dp_metadata = getattr(get_forward_context(), "dp_metadata", None)
            across = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
            if across is not None and across.numel():
                return max(num_tokens, int(across.max()))
        return num_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        if shared_experts is not None:
            raise ValueError(f"{self.prefix} fuses the shared expert into MegaMoE; pass shared_experts=None.")
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(f"MegaMoE got {num_tokens} tokens, but was sized for {self.max_num_tokens}.")
        self._synchronize_first_launch()
        runtime = self.get_symm_buffer()
        bucket = runtime.bucket(self._bucket_tokens(num_tokens))
        is_padding = None
        if is_forward_context_available():
            import vllm.envs as envs

            if envs.VLLM_MOE_SKIP_PADDING:
                is_padding = get_forward_context().is_padding
                if is_padding is not None:
                    is_padding = is_padding[:num_tokens]
        # Capture logical ids; staging maps them to EPLB physical replicas (as vLLM's native forward).
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if hidden_states.stride(-1) != 1 or hidden_states.stride(0) != self.hidden_size:
            hidden_states = hidden_states.contiguous()
        eplb = eplb_kernel_args(self.eplb_state)
        check_eplb_args(eplb, self.num_logical_experts, self.num_experts, self.prefix)
        y = runtime.run(
            hidden_states, topk_ids, topk_weights, is_padding, bucket=bucket, layer_kwargs=self._nvfp4_kwargs,
            norm=self._input_norm_const, scale_mul=self._scale_mul, eplb=eplb,
        )
        return y[:num_tokens]

    # ------------------------------------------------------------------------------------------------- FRONT2
    def front2_ready(self) -> bool:
        """FRONT2 serves this layer's next gate + call (decided once per layer: ``armed``, ``off`` or
        ``unavailable:<why>``), not while a routing capture needs the logical ids (FRONT2 writes physical ids only)."""
        if not self.__dict__.get("_front2_word"):
            self._front2_decide()
        return self.__dict__["_front2_fn"] is not None and self.capture_fn is None

    def _front2_decide(self) -> None:
        fn, word = None, "off"
        if front2_enabled() and not getattr(self, "_nvfp4", True):  # a class that also serves MXFP4 layers
            word = "not_admitted:not_nvfp4"
        elif front2_enabled():
            from flashinfer.moe_ep.sm107_nvfp4 import Nvfp4MegaMoERuntime
            from vllm.utils.deep_gemm import _import_deep_gemm, _lazy_init

            _lazy_init()  # the JIT cache dir and PDL setting bf16_mega_gate gets
            dg = _import_deep_gemm()
            fn = getattr(dg, "bf16_mega_gate_front2", None) if dg is not None else None
            if fn is None:
                word = "unavailable:deep_gemm_lacks_bf16_mega_gate_front2"
            elif not hasattr(Nvfp4MegaMoERuntime, "external_planes"):
                fn, word = None, "unavailable:flashinfer_lacks_external_planes"
            else:
                word = "armed"
            rank = get_ep_group().rank_in_group
            _front2_census["rank"] = rank
            logger.info("NVFP4 MegaMoE FRONT2 layer=%s rank=%d word=%s", self.prefix, rank, word)
            if not _front2_census["atexit"]:
                atexit.register(_front2_census_line, "exit")
                _front2_census["atexit"] = True
        object.__setattr__(self, "_front2_fn", fn)
        object.__setattr__(self, "_front2_word", word)
        _front2_census["layers"][word.split(":")[0]] += 1

    def forward_front2(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        gate_bias: torch.Tensor,
        top_k: int,
        *,
        scoring_func: str,
        routed_scaling_factor: float,
        ep_rank: int,
        image_bias: torch.Tensor | None = None,
        image_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The model's ``bf16_mega_gate`` and :meth:`forward` in one FRONT2 launch + the kernel launch (the same
        planes and outputs as gate -> staging -> kernel)."""
        if self.__dict__.get("_front2_fn") is None:
            word = self.__dict__.get("_front2_word") or "undecided"
            raise RuntimeError(f"{self.prefix}: FRONT2 called while not armed ({word})")
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(f"MegaMoE got {num_tokens} tokens, but was sized for {self.max_num_tokens}.")
        self._synchronize_first_launch()
        runtime = self.get_symm_buffer()
        bucket = runtime.bucket(self._bucket_tokens(num_tokens))
        is_padding = None
        if is_forward_context_available():
            import vllm.envs as envs

            if envs.VLLM_MOE_SKIP_PADDING:
                is_padding = get_forward_context().is_padding
                if is_padding is not None:
                    is_padding = is_padding[:num_tokens]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        eplb = eplb_kernel_args(self.eplb_state)
        check_eplb_args(eplb, self.num_logical_experts, self.num_experts, self.prefix)
        mapping, replicas, load, record, unpadded = eplb[:5]
        planes = runtime.external_planes(bucket)
        self._front2_fn(
            hidden_states, gate_weight, top_k, routed_scaling_factor, ep_rank, gate_bias,
            planes["act"], planes["act_sf"], planes["topk_idx"], planes["topk_weights"],
            scoring_func=scoring_func, image_bias=image_bias, image_token_mask=image_token_mask, format=0,
            norm_const=self._input_norm_const, scale_multiplier=self._scale_mul, is_padding=is_padding,
            logical_to_physical=mapping, replica_count=replicas, expert_load=load, record_enabled=record,
            num_unpadded_tokens=unpadded, xs_q=planes["shared_act"], xs_sf=planes["shared_act_sf"],
            staging_rule=FRONT2_STAGING_RULE,
        )
        _front2_census["calls"] += 1
        calls = _front2_census["calls"]
        if calls & (calls - 1) == 0 and calls <= 4096:
            _front2_census_line("running")
        y = runtime.launch(bucket, self._nvfp4_kwargs)
        return y[:num_tokens]


VllmNvfp4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _routed_names() -> tuple[str, ...]:
    return _ROUTED_NAMES


if os.environ.get("VLLM_NVFP4_MEGAMOE_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)
