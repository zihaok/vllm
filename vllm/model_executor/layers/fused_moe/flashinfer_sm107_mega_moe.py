# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's CuTe DSL SM107 MegaMoE through the native expert interface."""

from typing import Any, ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
)

logger = init_logger(__name__)


class FlashInferSM107MegaMoEExperts(DeepGemmMegaMoEExperts):
    """MXFP4 routed experts with fused MXFP8 shared experts on Rubin.

    The registered routed parameters own die-local allocations from construction,
    so checkpoint reloads preserve their addresses and graph-capture lifetime.
    Scales and shared weights retain the native MegaMoE layout.
    """

    _symm_buffers: ClassVar[dict[tuple, Any]] = {}
    _prepared_variants: ClassVar[set[tuple]] = set()
    _synchronized_ep_groups: ClassVar[set[tuple[int, int]]] = set()

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any):
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("FlashInfer SM107 MegaMoE does not support EPLB")
        if kwargs["weight_format"] != MEGA_MOE_MXFP4:
            raise NotImplementedError("FlashInfer SM107 MegaMoE requires MXFP4 weights")
        if kwargs["activation"].name != "swiglu":
            raise NotImplementedError("FlashInfer SM107 MegaMoE requires SwiGLU")
        backend = self._import_backend()
        if not backend.is_localization_available():
            raise NotImplementedError(
                "FlashInfer SM107 MegaMoE requires a two-domain SM107 GPU "
                "and cuda-python 13.4 or later"
            )
        super().__init__(vllm_config, **kwargs)
        logger.info_once(
            "Using FlashInfer CuTe DSL SM107 MegaMoE with die-local routed weights "
            "and %d fused shared expert(s).",
            self.num_shared_experts,
        )

    @staticmethod
    def _import_backend():
        import flashinfer.mega_moe_deepgemm as backend

        return backend

    def _create_weights(self) -> None:
        backend = self._import_backend()
        reason = backend.get_unsupported_reason_for_mega_moe(
            get_ep_group().world_size,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
            fast_math=self.fast_math,
        )
        if reason is not None:
            raise NotImplementedError(f"FlashInfer SM107 MegaMoE: {reason}")
        super()._create_weights()
        for name in ("w13_weight", "w2_weight"):
            param = getattr(self, name)
            param.data = backend.localize(param.data)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if param is not self.w13_weight and param is not self.w2_weight:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        local_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_ids or param_prefix not in weight_name:
            return False if return_success else None
        for local_id in local_ids:
            dst = param.data[:, local_id]
            logical = dst.flatten(0, 1).clone()
            self._load_weight(logical, loaded_weight, shard_id, False)
            dst.copy_(logical.view_as(dst))
        return True if return_success else None

    def get_symm_buffer(self):
        buffer = getattr(self, "_runtime_buffer", None)
        if buffer is not None:
            return buffer
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up FlashInfer SM107 MegaMoE before capture")
        buffer = super().get_symm_buffer()
        key = (
            id(get_ep_group().device_group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            buffer.num_max_tokens_per_rank,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
            self.activation,
            self.fast_math,
        )
        if key not in self._prepared_variants:
            backend = self._import_backend()
            backend.get_balanced_sm_locality_domains()
            backend.precompile_mega_moe_variants(
                get_ep_group().world_size,
                self.num_experts,
                buffer.num_max_tokens_per_rank,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.num_shared_experts,
                **self._kernel_kwargs,
                fast_math=self.fast_math,
                per_die=True,
                localized_weights=True,
            )
            torch.distributed.barrier(group=get_ep_group().cpu_group)
            self._prepared_variants.add(key)
        self._runtime_buffer = buffer
        return buffer

    def set_eplb_state(self, *args, **kwargs) -> None:
        raise NotImplementedError("FlashInfer SM107 MegaMoE does not support EPLB")

    def get_expert_weights(self) -> list[torch.Tensor]:
        raise NotImplementedError("FlashInfer SM107 MegaMoE does not support EPLB")


FlashInferSM107MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
