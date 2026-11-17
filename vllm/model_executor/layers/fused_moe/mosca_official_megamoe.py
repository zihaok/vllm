# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSCA: the official CuTe DSL MegaMoE (FlashInfer moe_ep, SM107) behind DeepSeek-V4's native MegaMoE entry.

Default off. ``VLLM_MOSCA_OFFICIAL_MEGAMOE=1`` makes ``DeepseekV4MoE`` build these experts instead of
``DeepGemmMegaMoEExperts`` (or an out-of-tree layer that replaces it). Everything before the experts is the
served model's own code: the gate (GateLinear + fused_topk_bias at small T, DeepGEMM bf16_mega_gate above)
produces logical top-k ids and weights, exactly as for those experts.

Routed format = DATA (``OFFICIAL_FORMATS``), one class and one code path for both:
  w4a8  the checkpoint's MXFP4 routed experts x MXFP8 E4M3 activations (per-32 E8M0); FlashInfer
        ``sm107_mxfp8_mxfp4_bf16_cutedsl``;
  w4a4  the checkpoint's NVFP4 routed experts (per-16 E4M3 + per-expert ``weight_scale_2`` / ``input_scale``)
        x NVFP4 activations; FlashInfer ``sm107_nvfp4_nvfp4_bf16_cutedsl``, epilogue scalars by vLLM main's rule
        (``sm107_nvfp4_epilogue``).
The layer's format is the checkpoint's, by the model's own per-layer rule (``checkpoint_weight_format``: the served quant
config's ``_is_nvfp4_expert_layer``, the rule its FusedMoE path uses), so one checkpoint can serve both formats (NVFP4
3431dde: target layers w4a4, the MTP drafter w4a8) and a FusedMoE prefill loads every layer in the same format.

Per call:
  1. EPLB: logical -> physical replica ids + load record (vLLM's ``eplb_map_to_physical_and_record``),
     padded tokens unrouted (``VLLM_MOE_SKIP_PADDING``), then int32;
  2. FlashInfer ``MoEEpMegaLayer.forward`` on a capacity bucket's workspace: FlashInfer's SM107 staging
     (the format's activations and the shared expert's MXFP8 input), the vendored cutedsl_megamoe kernel
     (gen-phase variant, the format's routed weights, FlashInfer's fused MXFP8 shared expert,
     ``Sm107SharedExpertConfig``) and its separate BF16 top-k reduce, which adds the shared expert's output.

Weights: the checkpoint's routed experts and MXFP8 shared expert are loaded in their canonical layout
(no requantization) and handed to FlashInfer's prequantized preprocessing once after loading (the shared
expert as the routed pack's ``shared_expert``, a one-expert ``PrequantizedMoEWeights``); the source
copies are released. EPLB moves the kernel-resident routed tensors in place (expert-major byte views; w4a4
adds the per-expert epilogue vectors, so global scales move with their experts).

Capacity buckets: the fused shared expert runs over the whole workspace capacity, so a call uses the
smallest bucket that holds the rank-uniform token key (the step's largest token count over the DP ranks;
under CUDA graphs every rank is padded to it). Buckets, variant and per-bucket knobs are DATA:
  VLLM_MOSCA_OFFICIAL_BUCKETS   comma list of capacities (default 32,64,128,256,512; the layer capacity is added)
  VLLM_MOSCA_OFFICIAL_VARIANT   genphase (default) | inference
  VLLM_MOSCA_OFFICIAL_KNOBS     optional JSON file {"k<top_k>/<bucket>": {knob: value}, "<bucket>": {...}, "*": {...}} (the
                                most specific key wins: per MoE geometry by its top-k, then per bucket, then all); absent =
                                FlashInfer's own defaults for the variant; the w4a8 layers' table
  VLLM_MOSCA_OFFICIAL_KNOBS_NVFP4  the same for the w4a4 layers (one table per format, each format's env in OFFICIAL_FORMATS)
  VLLM_MOSCA_OFFICIAL_SHARED    1 (default): fuse the shared expert; 0: the model runs its shared MLP serially
Census lines (WARNING, prefix ``MOSCA_OFFICIAL``) name every armed layer, every built workspace and the call
counts per bucket, on every decode rank.
"""

from __future__ import annotations

import atexit
import collections
import dataclasses
import json
import math
import os
import sys
import time
import types
from typing import Any, ClassVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
    _matches_weight_spec,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger("vllm.mosca.official_megamoe")

OFFICIAL_SWITCH = "VLLM_MOSCA_OFFICIAL_MEGAMOE"
KERNEL_SRC = "cutedsl_megamoe 70a6878d + export patches (FlashInfer PR #9426 4c891e24), FlashInfer shared expert"
_GLUE = "official-armA-v5"


def official_megamoe_enabled() -> bool:
    return os.environ.get(OFFICIAL_SWITCH, "0").strip() not in ("", "0")


# --------------------------------------------------------------------------------------------- FlashInfer import
def _flashinfer_moe_ep_compat() -> list[str]:
    """Names the vendored moe_ep (FlashInfer main) imports from an older installed FlashInfer.

    Only additions: a missing name is bound to moe_ep's own definition (``flashinfer/autotuner/protocol.py``
    copied with moe_ep) or to a plain exception class; nothing present is replaced. Returns the bound names.
    """
    bound: list[str] = []
    import flashinfer.autotuner as autotuner

    if not hasattr(autotuner, "TuningContext") or not hasattr(autotuner, "TuningResult"):
        import importlib

        protocol = importlib.import_module("flashinfer.autotuner.protocol")
        for name in ("TuningContext", "TuningResult", "TuningProtocol"):
            if not hasattr(autotuner, name) and hasattr(protocol, name):
                setattr(autotuner, name, getattr(protocol, name))
                bound.append(f"flashinfer.autotuner.{name}")
    import flashinfer.fused_moe.layer as fused_moe_layer

    if not hasattr(fused_moe_layer, "NoEligibleBackend"):

        class NoEligibleBackend(RuntimeError):
            """No configured backend supports the requested MoE configuration."""

        fused_moe_layer.NoEligibleBackend = NoEligibleBackend
        bound.append("flashinfer.fused_moe.layer.NoEligibleBackend")
    return bound


_FI: dict[str, Any] = {}


def _flashinfer_api() -> dict[str, Any]:
    if _FI:
        return _FI
    bound = _flashinfer_moe_ep_compat()
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MegaConfig, MoEEpMegaLayer, MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp4_bf16_cutedsl.config import (
        Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_expert import Sm107SharedExpertWeights
    from flashinfer.moe_ep.config import Sm107SharedExpertConfig
    from flashinfer.moe_ep.weights import PrequantizedMoEWeights

    import flashinfer.moe_ep as moe_ep

    _FI.update(
        BootstrapConfig=BootstrapConfig,
        FleetParams=FleetParams,
        MegaConfig=MegaConfig,
        MoEEpMegaLayer=MoEEpMegaLayer,
        MoEEpTensors=MoEEpTensors,
        Config=Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
        ConfigNvfp4=Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        SharedConfig=Sm107SharedExpertConfig,
        SharedPrepared=Sm107SharedExpertWeights,
        Prequantized=PrequantizedMoEWeights,
        moe_ep_file=moe_ep.__file__,
        compat_bound=bound,
    )
    logger.warning(
        "MOSCA_OFFICIAL import moe_ep=%s compat_bound=%s cute_dsl_arch=%s",
        moe_ep.__file__,
        ",".join(bound) or "none",
        os.environ.get("CUTE_DSL_ARCH", "unset"),
    )
    return _FI


# --------------------------------------------------------------------------------------------- formats (DATA)
@dataclasses.dataclass(frozen=True)
class OfficialFormat:
    """One routed format this path serves, keyed by the checkpoint's MegaMoE weight format name."""

    word: str  # census word
    quant_kind: str  # FlashInfer SM107 quant kind
    config: str  # key of the FlashInfer config class in _flashinfer_api()
    knobs_env: str  # env naming this format's knob table


OFFICIAL_FORMATS: dict[str, OfficialFormat] = {
    "mxfp4": OfficialFormat("w4a8", "mxfp4_mxfp8", "Config", "VLLM_MOSCA_OFFICIAL_KNOBS"),
    "nvfp4": OfficialFormat("w4a4", "nvfp4", "ConfigNvfp4", "VLLM_MOSCA_OFFICIAL_KNOBS_NVFP4"),
}


def official_format(name: str) -> OfficialFormat:
    fmt = OFFICIAL_FORMATS.get(name)
    if fmt is None:
        raise NotImplementedError(f"official_format_unserved:{name} (arm A serves {sorted(OFFICIAL_FORMATS)})")
    return fmt


def checkpoint_weight_format(vllm_config: VllmConfig, prefix: str) -> tuple[Any, str]:
    """The routed MegaMoE weight format of the MoE layer at ``prefix`` and its source word: the model's own per-layer rule.

    The served quant config (DSv4.1: ``DeepseekV4FP8Config``) decides each routed-expert layer's checkpoint format for the
    FusedMoE path with ``_is_nvfp4_expert_layer`` (``moe_quant_algo`` NVFP4 and the layer, MTP layers renamed ``mtp.<k>``, not
    in the ModelOpt ``ignore`` list) on ``expert_dtype`` fp4; this path reads the same rule, so a prefill server on the FusedMoE
    path and a decode server on this path load every layer in the same format (NVFP4 3431dde: target layers w4a4, the MTP drafter
    layers w4a8; the MXFP4 checkpoint dba1be0: every layer w4a8). A quant config without the rule types its absence in the
    source word (MXFP4 on expert_dtype fp4); another expert dtype raises ``official_format_unserved:expert_dtype_<d>``.
    """
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import MEGA_MOE_WEIGHT_FORMATS

    experts = prefix if prefix.endswith(".experts") else f"{prefix}.experts"
    qc = getattr(vllm_config, "quant_config", None)
    hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    dtype = getattr(qc, "expert_dtype", None) or getattr(hf, "expert_dtype", None) or "fp4"
    if dtype != "fp4":
        raise NotImplementedError(f"official_format_unserved:expert_dtype_{dtype}")
    rule = getattr(qc, "_is_nvfp4_expert_layer", None)
    if rule is None:
        algo, source = "mxfp4", f"absent:{type(qc).__name__}._is_nvfp4_expert_layer:expert_dtype_fp4"
    else:
        algo = "nvfp4" if rule(experts) else "mxfp4"
        source = f"{type(qc).__name__}._is_nvfp4_expert_layer"
    official_format(algo)
    return MEGA_MOE_WEIGHT_FORMATS[algo], source


@dataclasses.dataclass
class OfficialEpilogue:
    """NVFP4 (w4a4) epilogue constants of one layer: the activation input scale's reciprocal (staging) and the per-local-
    expert fp32 vectors the kernel reads (fc1 / fc2 alphas, FC1-output quantization constant). EPLB moves the vectors."""

    input_norm_const: float
    fc1_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc1_norm_const: torch.Tensor
    w13_amax: float
    w2_amax: float

    def tensors(self) -> dict[str, torch.Tensor]:
        return dict(fc1_alpha=self.fc1_alpha, fc2_alpha=self.fc2_alpha, fc1_norm_const=self.fc1_norm_const)


def _require_finite_positive(name: str, t: torch.Tensor) -> None:
    if not bool(((t > 0) & torch.isfinite(t)).all()):
        raise ValueError(f"official_nvfp4_scale_invalid:{name} (finite positive per expert required)")


def sm107_nvfp4_epilogue(
    w13_weight_scale_2: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    process_group: Any = None,
) -> OfficialEpilogue:
    """vLLM main's rule (2f5dc1d7 ``fused_moe/experts/flashinfer_moe_ep.py`` ``epilogue_from_quant_config`` +
    ``sm107_nvfp4_epilogue``): fc1 / fc2 alphas = the per-expert weight global scales (``weight_scale_2``) times the layer
    maximum of the checkpoint's activation input scales over every EP rank (tokens are quantized on their source rank),
    input_norm_const = 1 / w13 amax, fc1_norm_const = 1 / w2 amax. Collective over ``process_group`` (every EP rank, same
    layers, same order). The alphas are scaled in place (they stay the layer's ``weight_scale_2`` storage)."""
    _require_finite_positive("w13_weight_scale_2", w13_weight_scale_2)
    _require_finite_positive("w2_weight_scale_2", w2_weight_scale_2)
    if w13_weight_scale_2.dim() != 1 or w2_weight_scale_2.dim() != 1:
        raise ValueError("official_nvfp4_scale_invalid:one weight global scale per expert required")
    amax = torch.stack((w13_input_scale.amax(), w2_input_scale.amax())).float()
    finite = torch.stack((torch.isfinite(w13_input_scale).all(), torch.isfinite(w2_input_scale).all()))
    amax = torch.where(finite, amax, torch.full_like(amax, math.inf))
    if process_group is not None and torch.distributed.get_world_size(process_group) > 1:
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=process_group)
    w13_amax, w2_amax = amax.tolist()
    if not (0.0 < w13_amax < math.inf and 0.0 < w2_amax < math.inf):
        raise ValueError(
            f"official_nvfp4_scale_invalid:input_scale amax w13={w13_amax} w2={w2_amax} (finite positive over the EP group)"
        )
    fc1 = w13_weight_scale_2.float().contiguous()
    fc2 = w2_weight_scale_2.float().contiguous()
    if fc1.data_ptr() != w13_weight_scale_2.data_ptr() or fc2.data_ptr() != w2_weight_scale_2.data_ptr():
        raise ValueError("official_nvfp4_scale_invalid:weight_scale_2 must be contiguous fp32 (scaled in place)")
    fc1.mul_(w13_amax)
    fc2.mul_(w2_amax)
    return OfficialEpilogue(
        input_norm_const=1.0 / w13_amax,
        fc1_alpha=fc1,
        fc2_alpha=fc2,
        fc1_norm_const=torch.full_like(fc2, 1.0 / w2_amax),
        w13_amax=w13_amax,
        w2_amax=w2_amax,
    )


# --------------------------------------------------------------------------------------------- data
def _buckets(capacity: int) -> list[int]:
    raw = os.environ.get("VLLM_MOSCA_OFFICIAL_BUCKETS", "32,64,128,256,512")
    out = sorted({int(x) for x in raw.replace(" ", "").split(",") if x} | {int(capacity)})
    return [b for b in out if 0 < b <= capacity]


def _knob_table(env: str = "VLLM_MOSCA_OFFICIAL_KNOBS") -> dict[str, dict]:
    path = os.environ.get(env, "").strip()
    if not path:
        return {}
    with open(path) as f:
        table = json.load(f)
    out: dict[str, dict] = {}
    for key, knobs in table.items():
        fixed = {}
        for name, value in knobs.items():
            fixed[name] = tuple(value) if isinstance(value, list) else value
        out[str(key)] = fixed
    return out


# --------------------------------------------------------------------------------------------- census
_census = {
    "calls": collections.Counter(),
    "keys": collections.Counter(),
    "formats": collections.Counter(),
    "armed": 0,
    "workspaces": set(),
    "n": 0,
    "atexit": False,
    "last_t": 0.0,
}


def _census_summary(tag: str) -> None:
    calls = ",".join(f"{k}:{v}" for k, v in sorted(_census["calls"].items(), key=lambda x: str(x[0])))
    keys = ",".join(f"{k}:{v}" for k, v in sorted(_census["keys"].items()))
    formats = ",".join(f"{k}:{v}" for k, v in sorted(_census["formats"].items()))
    logger.warning(
        "MOSCA_OFFICIAL calls tag=%s rank=%s armed=%d workspaces=%d n=%d per_bucket=%s key_words=%s glue=%s formats=%s",
        tag,
        _census.get("rank", "?"),
        _census["armed"],
        len(_census["workspaces"]),
        _census["n"],
        calls or "none",
        keys or "none",
        _GLUE,
        formats or "none",
    )


def _census_call(geometry: str, bucket: int, word: str) -> None:
    _census["calls"][f"{geometry}/{bucket}"] += 1
    _census["keys"][word] += 1
    _census["n"] += 1
    n = _census["n"]
    now = time.monotonic()
    if (n & (n - 1)) == 0 and n <= 4096 or n % 4096 == 0 or now - _census["last_t"] >= 600.0:
        _census_summary("running")
    _census["last_t"] = now
    if not _census["atexit"]:
        atexit.register(_census_summary, "exit")
        _census["atexit"] = True


def _rank_uniform_key(num_tokens: int) -> tuple[int, str]:
    """The call's token key every EP rank shares: the DP maximum of the step, else this rank's count (typed)."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return num_tokens, "local:no_forward_context"
        dpm = get_forward_context().dp_metadata
        if dpm is None:
            return num_tokens, "local:no_dp_metadata"
        key = int(dpm.num_tokens_across_dp_cpu.max())
    except Exception as exc:  # noqa: BLE001 - a typed absence, the local count is used
        return num_tokens, f"local:exception:{type(exc).__name__}"
    if key < num_tokens:
        return num_tokens, "local:own_rows_above_key"
    return key, "dp_max"


# --------------------------------------------------------------------------------------------- FlashInfer layer
def build_official_layer(
    *,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    shared: dict[str, torch.Tensor] | None,
    num_experts: int,
    num_local_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    shared_intermediate: int | None,
    clamp: float | None,
    variant: str,
    max_tokens: int,
    ep_world: int,
    ep_rank: int,
    process_group: Any,
    device_index: int,
    weight_format: str = "mxfp4",
    epilogue: OfficialEpilogue | None = None,
) -> tuple[Any, list[torch.Tensor]]:
    """One FlashInfer MoEEpMegaLayer over canonical routed (+ MXFP8 shared) weights in ``weight_format``.

    ``w13`` [E_local, 2I, H/2] packed E2M1 (all gate rows, then all up rows) with ``w13_scale`` [E_local, 2I, H/g]
    block scales (mxfp4: g 32, E8M0; nvfp4: g 16, E4M3); ``w2`` [E_local, H, I/2] with ``w2_scale`` [E_local, H, I/g];
    ``shared`` one MXFP8 expert (keys w13, w2, w13_scale, w2_scale; E4M3 data, E8M0 per-32 scales); ``epilogue`` the
    w4a4 constants (``sm107_nvfp4_epilogue``; required for nvfp4, absent for mxfp4). Returns the layer and the routed
    kernel tensors as ``(E_local, -1)`` byte views (fc1 weight, fc1 scale, fc2 weight, fc2 scale; w4a4 adds fc1_alpha,
    fc2_alpha, fc1_norm_const) for in-place EPLB moves.
    """
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import MEGA_MOE_WEIGHT_FORMATS

    fmt = official_format(weight_format)
    if (epilogue is None) == MEGA_MOE_WEIGHT_FORMATS[weight_format].routed.has_global_scale:
        raise ValueError(f"official_epilogue_mismatch:{weight_format} (w4a4 needs its epilogue, w4a8 has none)")
    fi = _flashinfer_api()
    u8, e8 = torch.uint8, torch.float8_e8m0fnu
    sfd = MEGA_MOE_WEIGHT_FORMATS[weight_format].routed.scale_dtype
    routed = dict(w13=w13.view(u8), w2=w2.view(u8), w13_scale=w13_scale.view(sfd), w2_scale=w2_scale.view(sfd))
    shared_pack = None
    if shared is not None:
        shared_pack = fi["Prequantized"](
            w13=shared["w13"].unsqueeze(0),
            w2=shared["w2"].unsqueeze(0),
            w13_scale=shared["w13_scale"].view(e8).unsqueeze(0),
            w2_scale=shared["w2_scale"].view(e8).unsqueeze(0),
        )
    weights = fi["Prequantized"](**routed, shared_expert=shared_pack)
    config = fi[fmt.config](
        intermediate_size=intermediate,
        top_k=top_k,
        gate_up_clamp=None if clamp is None else float(clamp),
        fast_math=True,
        in_kernel_fc2_reduce=False,
        apply_routing_weights_before_fc2=True,
        kernel_variant=variant,
        cluster_shape_mn=(4, 1),
        fc2_use_bulk=True,
        combine_dtype="bf16",
        shared_expert=None if shared is None else fi["SharedConfig"](intermediate_size=shared_intermediate),
        **({} if epilogue is None else dict(input_norm_const=epilogue.input_norm_const)),
    )
    bootstrap = fi["BootstrapConfig"](
        world_size=ep_world, rank=ep_rank, process_group=process_group, device=device_index
    )
    fleet = fi["FleetParams"](num_experts=num_experts, max_tokens_per_rank=max_tokens, token_hidden_size=hidden)
    mega = fi["MoEEpMegaLayer"](
        bootstrap, fleet, weights, fi["MegaConfig"](megakernel=config, quantize_input=True, preprocess_weights=True)
    )
    transformed = mega._get_transformed_weights()
    routed_pair = transformed.routed if isinstance(transformed, fi["SharedPrepared"]) else transformed
    views = [_expert_major_bytes(t, num_local_experts) for pair in routed_pair for t in pair]
    if epilogue is not None:
        views += [_expert_major_bytes(t, num_local_experts) for t in epilogue.tensors().values()]
    return mega, views


def create_bucket_workspace(
    mega: Any, bucket: int, top_k: int | None = None, knobs_env: str = "VLLM_MOSCA_OFFICIAL_KNOBS"
) -> tuple[Any, dict[str, Any]]:
    """A capacity-``bucket`` workspace of ``mega`` (pooled across layers of one geometry) and its census fields; the knobs
    come from the table ``knobs_env`` names (one table per format)."""
    table = _knob_table(knobs_env)
    knobs = table.get(f"k{top_k}/{bucket}", table.get(str(bucket), table.get("*")))
    handle = mega.create_workspace(bucket, knobs=dict(knobs) if knobs else None)
    backend = handle._backend_workspace
    cfg = getattr(backend, "config", None)
    kname = getattr(getattr(backend, "kernel", None), "name", None)
    kname = kname() if callable(kname) else kname
    info = dict(
        capacity=getattr(cfg, "padded_tokens_per_rank", "?"),
        variant=getattr(cfg, "kernel_variant", "?"),
        tiler=getattr(cfg, "resolved_mma_tiler_mnk", "?"),
        cluster=getattr(cfg, "cluster_shape_mn", "?"),
        schedule=getattr(cfg, "schedule_policy", "?"),
        work_id=getattr(cfg, "work_id_mode", "?"),
        fc2_bulk=getattr(cfg, "fc2_use_bulk", "?"),
        shared=getattr(getattr(cfg, "shared_expert", None), "kind", None),
        quant=getattr(cfg, "quant_kind", "?"),
        knobs="table" if knobs else "flashinfer_default",
        kernel=str(kname),
    )
    return handle, info


def official_forward(
    mega: Any,
    handle: Any,
    hidden_states: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    epilogue: OfficialEpilogue | None = None,
):
    """Stage (quantize + routes [+ shared input]) and run the kernel on ``handle``; returns BF16 routed + shared. w4a4
    passes its per-expert epilogue vectors per call (vLLM main's MoEEpTensors fields; FlashInfer binds them)."""
    extra = {} if epilogue is None else epilogue.tensors()
    tensors = _FI["MoEEpTensors"](hidden_states=hidden_states, topk_ids=ids, topk_weights=weights, **extra)
    return mega.forward(tensors, workspace=handle)


# --------------------------------------------------------------------------------------------- post-load hook
class _OfficialPostLoad(QuantizeMethodBase):
    """Runs once after the checkpoint is loaded: hands the weights to FlashInfer (vLLM's post-load loop)."""

    requires_device_loading = False

    def create_weights(self, layer: nn.Module, *weight_args, **extra_weight_attrs):
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer._build_official()


# --------------------------------------------------------------------------------------------- experts
class MoscaOfficialMegaMoEExperts(DeepGemmMegaMoEExperts):
    """Routed experts (+ the fused shared expert) on the official CuTe DSL MegaMoE through FlashInfer."""

    _built_workspaces: ClassVar[dict[tuple, str]] = {}

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any):
        weight_format = kwargs["weight_format"]
        self._official = official_format(weight_format.name)
        if kwargs["activation"].name != "swiglu":
            raise NotImplementedError("arm A needs SwiGLU")
        # The model passes the checkpoint's format (checkpoint_weight_format); a model that passes another is refused.
        ckpt_format, self._format_source = checkpoint_weight_format(vllm_config, kwargs.get("prefix", ""))
        if ckpt_format.name != weight_format.name:
            raise ValueError(
                f"official_format_mismatch:{kwargs.get('prefix', '')}:model={weight_format.name}:"
                f"checkpoint={ckpt_format.name}({self._format_source})"
            )
        self._fuse_shared = os.environ.get("VLLM_MOSCA_OFFICIAL_SHARED", "1").strip() not in ("", "0")
        super().__init__(vllm_config, **kwargs)
        ep = get_ep_group()
        self._ep_rank, self._ep_world = ep.rank_in_group, ep.world_size
        _census["rank"] = f"{self._ep_rank}/{self._ep_world}"
        self._variant = os.environ.get("VLLM_MOSCA_OFFICIAL_VARIANT", "genphase").strip()
        self._bucket_list = _buckets(self.max_num_tokens)
        self._mega: Any = None
        self._handles: dict[int, Any] = {}
        self._kernel_views: list[torch.Tensor] | None = None
        self._geometry = f"E{self.num_experts}k{self.top_k}"
        self.quant_method = _OfficialPostLoad()

    # ---- shared expert: fused when the shared MLP is one unsharded MXFP8 expert of a supported width
    def _fusable_shared_experts(self, shared_experts: nn.Module | None) -> tuple[int, MegaMoEWeightSpec | None]:
        if shared_experts is None or not getattr(self, "_fuse_shared", True):
            return 0, None
        gate_up, down = shared_experts.gate_up_proj, shared_experts.down_proj
        spec = self.weight_format.shared[0]
        shared_size = down.weight.shape[-1]
        ok = (
            tuple(gate_up.weight.shape) == (2 * shared_size, self.hidden_size)
            and tuple(down.weight.shape) == (self.hidden_size, shared_size)
            and all(_matches_weight_spec(lin, spec) for lin in (gate_up, down))
            and shared_size % 128 == 0
        )
        if not ok:
            logger.warning(
                "MOSCA_OFFICIAL shared_unfused layer=%s reason=shape_or_format gate_up=%s down=%s",
                self.prefix,
                tuple(gate_up.weight.shape),
                tuple(down.weight.shape),
            )
            return 0, None
        # One fused shared expert of width shared_size (FlashInfer Sm107SharedExpertConfig.intermediate_size).
        self._shared_width = shared_size
        return 1, spec

    # ---- canonical (checkpoint) layout: [E, rows, K / values_per_element] data and [E, rows, K / group] scale bytes;
    #      w4a4 adds the checkpoint's per-expert fp32 global scales (``weight_scale_2``, ``input_scale``)
    def _create_weights(self) -> None:
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        fmt = self.weight_format
        self._register_canonical("w13", (e, 2 * i, h), fmt.routed)
        self._register_canonical("w2", (e, h, i), fmt.routed)
        if fmt.routed.has_global_scale:
            for name in ("w13_weight_scale_2", "w13_input_scale", "w2_weight_scale_2", "w2_input_scale"):
                self.register_parameter(name, nn.Parameter(torch.zeros(e, dtype=torch.float32), requires_grad=False))
        if self.shared_spec is not None:
            s = self._shared_width
            self._register_canonical("shared_l1", (2 * s, h), self.shared_spec)
            self._register_canonical("shared_l2", (h, s), self.shared_spec)

    def _register_canonical(self, name: str, shape: tuple[int, ...], spec: MegaMoEWeightSpec) -> None:
        *outer, n, k = shape
        data = torch.zeros(*outer, n, k // spec.values_per_element, dtype=spec.dtype)
        self.register_parameter(f"{name}_weight", nn.Parameter(data, requires_grad=False))
        if spec.scale_group_k is not None:
            g = spec.scale_group_k
            if k % g:
                raise ValueError(f"{name}: K {k} is not a multiple of the {g}-wide scale block")
            scale = torch.zeros(*outer, n, k // g, dtype=torch.uint8)
            self.register_parameter(f"{name}_weight_scale", nn.Parameter(scale, requires_grad=False))

    def _record_global_scale(self, weight_name: str, scale: torch.Tensor, shard_id: str, expert_id: int) -> None:
        """The parent's w4a4 record (w1 / w3 ``weight_scale_2`` must match: one fc1 alpha per expert) plus the same check
        on w1 / w3 ``input_scale`` (one FC1 input grid per expert; both write the expert's ``w13_input_scale`` slot)."""
        super()._record_global_scale(weight_name, scale, shard_id, expert_id)
        if not weight_name.endswith("w13_input_scale"):
            return
        pending = self.__dict__.setdefault("_pending_l1_input_scales", {})
        prev = pending.pop(expert_id, None)
        value = float(scale)
        if prev is None or prev[0] == shard_id:
            pending[expert_id] = (shard_id, value)
        elif not math.isclose(value, prev[1], rel_tol=1e-5):
            raise ValueError(
                f"official_nvfp4_scale_invalid:expert {expert_id} {prev[0]} input_scale {prev[1]} != {shard_id} {value}"
            )

    def _load_weight(self, dst: torch.Tensor, src: torch.Tensor, shard_id: str, is_scale: bool) -> None:
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
        cols = dst.shape[1]
        if is_scale:
            sf = src.view(torch.uint8)
            if rows % sf.shape[0] or cols % sf.shape[1]:
                raise ValueError(f"scale {tuple(sf.shape)} does not tile ({rows}, {cols})")
            src = sf.repeat_interleave(rows // sf.shape[0], 0).repeat_interleave(cols // sf.shape[1], 1)
        else:
            src = src.view(dst.dtype)
            if tuple(src.shape) != (rows, cols):
                raise ValueError(f"{shard_id} weight {tuple(src.shape)} != ({rows}, {cols})")
        if shard_id == "w2":
            dst.copy_(src)
        else:  # all gate rows (w1), then all up rows (w3): FlashInfer's canonical FC1 order
            half = 0 if shard_id == "w1" else 1
            dst[half * rows : (half + 1) * rows].copy_(src)

    # ---- after loading: FlashInfer owns the kernel layout; EPLB moves those tensors
    def _build_official(self) -> None:
        if self._mega is not None:
            return
        ep = get_ep_group()
        shared = None
        if self.has_fused_shared_experts:
            shared = dict(
                w13=self.shared_l1_weight.data,
                w2=self.shared_l2_weight.data,
                w13_scale=self.shared_l1_weight_scale.data,
                w2_scale=self.shared_l2_weight_scale.data,
            )
        clamp = self.activation.clamp
        t0 = time.monotonic()
        self._epilogue = None
        if self.weight_format.routed.has_global_scale:
            self._epilogue = sm107_nvfp4_epilogue(
                self.w13_weight_scale_2.data,
                self.w2_weight_scale_2.data,
                self.w13_input_scale.data,
                self.w2_input_scale.data,
                ep.device_group,
            )
        self._mega, self._kernel_views = build_official_layer(
            w13=self.w13_weight.data,
            w2=self.w2_weight.data,
            w13_scale=self.w13_weight_scale.data,
            w2_scale=self.w2_weight_scale.data,
            shared=shared,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            top_k=self.top_k,
            hidden=self.hidden_size,
            intermediate=self.intermediate_size,
            shared_intermediate=getattr(self, "_shared_width", None),
            clamp=clamp,
            variant=self._variant,
            max_tokens=max(self._bucket_list),
            ep_world=ep.world_size,
            ep_rank=ep.rank_in_group,
            process_group=ep.device_group,
            device_index=torch.accelerator.current_device_index(),
            weight_format=self.weight_format.name,
            epilogue=self._epilogue,
        )
        # The canonical copies were FlashInfer's input only; release them (the layer keeps no source pack). The w4a4
        # weight_scale_2 storage IS the epilogue's alphas (scaled in place) and stays.
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                     "shared_l1_weight", "shared_l2_weight", "shared_l1_weight_scale", "shared_l2_weight_scale",
                     "w13_input_scale", "w2_input_scale"):
            param = getattr(self, name, None)
            if isinstance(param, nn.Parameter):
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        del shared
        _census["armed"] += 1
        _census["formats"][self._official.word] += 1
        epi = self._epilogue
        logger.warning(
            "MOSCA_OFFICIAL armed layer=%s rank=%d/%d experts=%d local=%d topk=%d hidden=%d inter=%d shared=%s "
            "fmt=%s variant=%s buckets=%s n_armed=%d build_s=%.1f src=%s format=%s fmt_src=%s epilogue=%s",
            self.prefix,
            ep.rank_in_group,
            ep.world_size,
            self.num_experts,
            self.num_local_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            f"fused:{self._shared_width}" if self.has_fused_shared_experts else "none",
            self._official.quant_kind,
            self._variant,
            ",".join(map(str, self._bucket_list)),
            _census["armed"],
            time.monotonic() - t0,
            KERNEL_SRC,
            self._official.word,
            self._format_source,
            "none" if epi is None else f"w13_amax:{epi.w13_amax:.6g},w2_amax:{epi.w2_amax:.6g}",
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        """Kernel-resident routed tensors as ``(num_local_experts, -1)`` byte views (EPLB moves rows in place); w4a4 adds
        the per-expert epilogue vectors (fc1_alpha, fc2_alpha, fc1_norm_const), so global scales move with their experts."""
        if self._kernel_views is None:
            raise RuntimeError(f"{self.prefix}: EPLB asked for expert weights before the official build")
        return list(self._kernel_views)

    # ---- the DeepGemmMegaMoEExperts interface its outside consumers read (vLLM's MegaMoE warmup:
    #      ``get_block_m`` keys its whole-layer runs, ``get_symm_buffer().x.device`` places its inputs)
    def _bucket_for(self, key: int) -> int:
        """The capacity bucket a call with token key ``key`` runs in (the one rule forward and get_block_m share)."""
        bucket = next((b for b in self._bucket_list if b >= key), None)
        if bucket is None:
            raise ValueError(f"{self.prefix}: {key} tokens exceed the largest bucket {self._bucket_list[-1]}")
        return bucket

    def get_block_m(self, num_tokens: int) -> int:
        """This path's call shape for ``num_tokens`` tokens: its capacity bucket (the warmup runs each bucket's largest M)."""
        return self._bucket_for(num_tokens)

    def get_symm_buffer(self) -> "_NoDeepGemmBuffer":
        """This path has no DeepGEMM symmetric buffer; consumers get the device and capacity, any other field raises."""
        if self._kernel_views:
            device = self._kernel_views[0].device
        else:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        return _NoDeepGemmBuffer(device, self._bucket_list[-1])

    # ---- workspaces
    def _handle(self, bucket: int) -> Any:
        handle = self._handles.get(bucket)
        if handle is not None:
            return handle
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MOSCA_OFFICIAL {self.prefix}: bucket {bucket} was not warmed before CUDA graph capture"
            )
        handle, info = create_bucket_workspace(self._mega, bucket, self.top_k, self._official.knobs_env)
        self._handles[bucket] = handle
        key = (self._geometry, bucket)
        if key not in self._built_workspaces:
            self._built_workspaces[key] = info["kernel"]
            _census["workspaces"].add(key)
            logger.warning(
                "MOSCA_OFFICIAL workspace rank=%d geometry=%s bucket=%d %s",
                self._ep_rank,
                self._geometry,
                bucket,
                " ".join(f"{k}={v}" for k, v in info.items()).replace(" ", " "),
            )
        return handle

    def _warm_all_buckets(self) -> None:
        """Compile and launch every bucket once, eagerly, before any capture (collective, same order on all ranks)."""
        for bucket in self._bucket_list:
            self._mega.warmup(workspace=self._handle(bucket))
        self._warmed = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        if shared_experts is not None and self.has_fused_shared_experts:
            raise ValueError(f"{self.prefix} fuses the shared expert; pass shared_experts=None.")
        if self._mega is None:
            raise RuntimeError(f"{self.prefix}: official experts used before the post-load build")
        if not getattr(self, "_warmed", False):
            self._warm_all_buckets()
        num_tokens = hidden_states.shape[0]
        key, word = _rank_uniform_key(num_tokens)
        bucket = self._bucket_for(key)
        _census_call(self._geometry, bucket, word)

        ids = topk_ids
        state = self.eplb_state
        if state is not None and state.logical_to_physical_map is not None:
            from vllm.model_executor.layers.fused_moe.router.base_router import eplb_map_to_physical_and_record
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id

            unpadded = state.num_unpadded_tokens_tensors
            ids = eplb_map_to_physical_and_record(
                topk_ids,
                state.expert_load_view,
                state.logical_to_physical_map,
                state.logical_replica_count,
                state.should_record_tensor,
                None if unpadded is None else unpadded[dbo_current_ubatch_id()],
            )
        import vllm.envs as envs
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                ids = ids.masked_fill(is_padding[:num_tokens].to(torch.bool).view(-1, 1), -1)
        ids = ids.to(torch.int32)
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        y = official_forward(
            self._mega, self._handles[bucket], hidden_states, ids, topk_weights.to(torch.float32), self._epilogue
        )
        if shared_experts is not None:
            y = y + shared_experts(hidden_states)
        return y


class _NoDeepGemmBuffer:
    """Stands where a DeepGEMM MegaMoE symmetric buffer is read: ``x`` (an empty tensor on the layer's device) and
    ``num_max_tokens_per_rank`` (the layer's capacity). Any other field is a typed AttributeError."""

    __slots__ = ("x", "num_max_tokens_per_rank")

    def __init__(self, device: torch.device, capacity: int):
        self.x = torch.empty(0, device=device, dtype=torch.bfloat16)
        self.num_max_tokens_per_rank = int(capacity)

    def __getattr__(self, name: str):
        raise AttributeError(f"arm A has no DeepGEMM symmetric buffer: field {name!r} does not exist (MOSCA official)")


def _expert_major_bytes(tensor: torch.Tensor, experts: int) -> torch.Tensor:
    """Contiguous expert-major storage of a kernel tensor as a ``(experts, -1)`` uint8 view (in place)."""
    if not tensor.is_contiguous():
        tensor = tensor.transpose(1, 2)
    if not tensor.is_contiguous() or tensor.shape[0] != experts:
        raise ValueError(f"kernel tensor is not expert-major: shape={tuple(tensor.shape)} stride={tensor.stride()}")
    return tensor.view(torch.uint8).view(experts, -1)


MoscaOfficialMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
