# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder-side SWA bounded replay: running the replay layers on their batch.

The layers past the last KV-source layer own nothing but sliding-window KV, so
in eager prefill steps they run on each request's last ``window`` rows only.
``DeepseekV41ModelState`` prepares those rows as a sub-batch with attention
metadata and a forward context of its own, like a microbatch;
``DecoderReplayLayers`` gathers the layer inputs by its rows, runs the layers
under that context and scatters the outputs back to batch rows.

Piecewise CUDA-graph steps (``VLLM_DSV41_GRAPH_BOUNDED_REPLAY``) do the same
through a seam: the batch's graph breaks at the replay layers, which run as a
graph of their own, captured once per capture size in a private pool on
persistent input, output and metadata buffers. A step replays the graph sized
for its replay rows -- each prefill's last window plus every decode -- or for
the whole batch when nothing trims. FULL graphs keep the layers inline.
"""

import bisect
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    BreakableCUDAGraphWrapper,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.utils.torch_utils import weak_ref_tensor, weak_ref_tensors


class DecoderReplayLayers:
    """Runs the replay layers on the step's replay batch.

    ``run_layers`` takes a batch's layer inputs and returns its per-row
    outputs. ``row_buffers`` hold per-row results the source layer's indexer
    publishes for the layers after it; they are compacted to the replay rows
    in place.
    """

    def __init__(
        self,
        window: int,
        run_layers: Callable[..., tuple[torch.Tensor, ...]],
        row_buffers: list[torch.Tensor],
    ) -> None:
        self.window = window
        self.run_layers = run_layers
        self.row_buffers = row_buffers
        # The replay batch, set by the model state every step: its rows of the
        # batch and its forward context. None runs the layers on the batch.
        self.rows: torch.Tensor | None = None
        self.forward_context: ForwardContext | None = None

        # Graph seam (piecewise steps). The model state sets, every piecewise
        # step, the replay batch's forward context built on persistent buffers
        # and its real / graph token counts; rows stays None when nothing trims.
        self.graph_enabled = envs.VLLM_DSV41_GRAPH_BOUNDED_REPLAY
        self.max_num_tokens = 0
        self.graph_forward_context: ForwardContext | None = None
        self.graph_num_tokens = 0
        self.graph_num_tokens_padded = 0
        self._graphs: dict[int, BreakableCUDAGraphCapture] = {}
        self._graph_outputs: dict[int, tuple[torch.Tensor, ...]] = {}
        self._graph_sizes: list[int] = []
        # Private pool of the replay graphs; memory profiling swaps it, like a
        # graph wrapper's, for its throwaway pool.
        self.graph_pool = None
        self._in_bufs: list[torch.Tensor | None] | None = None
        self._out_bufs: list[torch.Tensor] | None = None
        self._padding: torch.Tensor | None = None
        self._arange: torch.Tensor | None = None
        if self.graph_enabled:
            # Dropped with the batch graphs (e.g. after the memory-profiling
            # capture, whose KV cache the replay graphs would still point at).
            BreakableCUDAGraphWrapper._all_instances.add(self)  # type: ignore[arg-type]

    # --- graph seam ----------------------------------------------------------

    def clear_graphs(self) -> None:
        self._graphs.clear()
        self._graph_outputs.clear()
        self._graph_sizes.clear()

    def has_graph(self, num_tokens: int) -> bool:
        return num_tokens in self._graphs

    def graph_size_for(self, num_tokens: int) -> int:
        """The smallest captured replay graph holding ``num_tokens`` rows."""
        i = bisect.bisect_left(self._graph_sizes, num_tokens)
        assert i < len(self._graph_sizes), (num_tokens, self._graph_sizes)
        return self._graph_sizes[i]

    def _seam_applies(self, hidden_states: torch.Tensor | MoEOutput) -> bool:
        if not self.graph_enabled or not isinstance(hidden_states, torch.Tensor):
            return False
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return False
        return (
            is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode
            == CUDAGraphMode.PIECEWISE
            and self.graph_forward_context is not None
        )

    def _seam(
        self, num_tokens: int, inputs: tuple[torch.Tensor | None, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Run the replay layers' graph for the batch's ``num_tokens`` rows:
        capture it on first use, then replay it sized for this step's rows."""
        context = self.graph_forward_context
        assert context is not None, "piecewise step without a replay context"
        if num_tokens not in self._graphs:
            assert self.rows is None, "replay rows set during capture"
            return self._capture(num_tokens, inputs, context)

        rows = self.rows
        n = self.graph_num_tokens
        size = self.graph_num_tokens_padded
        graph = self._graphs[size]
        assert self._in_bufs is not None and self._out_bufs is not None
        if rows is None:
            assert size == num_tokens
            for buf, x in zip(self._in_bufs, inputs):
                if x is not None and buf is not None:
                    buf[:num_tokens].copy_(x)
        else:
            for buf in self.row_buffers:
                buf[:n].copy_(buf.index_select(0, rows))
            for buf, x in zip(self._in_bufs, inputs):
                if x is not None and buf is not None:
                    torch.index_select(x, 0, rows, out=buf[:n])
        self._set_padding(n, size)
        self._inherit_batch_context(context)
        with override_forward_context(context):
            graph.replay()
        outputs = self._graph_outputs[size]
        for out, src in zip(self._out_bufs, outputs):
            if rows is None:
                out[:num_tokens].copy_(src[:num_tokens])
            else:
                # The trimmed rows' outputs stay zero; nothing reads them.
                out[:num_tokens].zero_()
                out[:num_tokens].index_copy_(0, rows, src[:n])
        return tuple(out[:num_tokens] for out in self._out_bufs)

    @staticmethod
    def _inherit_batch_context(context: ForwardContext) -> None:
        # The replay graph is a piece of the batch's piecewise graph: ops that
        # pick graph-safe paths by mode (e.g. mHC stream overlap only in FULL
        # graphs) must see PIECEWISE, and DP-aware ones the batch's metadata.
        batch = get_forward_context()
        context.cudagraph_runtime_mode = batch.cudagraph_runtime_mode
        context.batch_descriptor = batch.batch_descriptor
        context.dp_metadata = batch.dp_metadata

    def _set_padding(self, num_real: int, size: int) -> None:
        assert self._padding is not None and self._arange is not None
        torch.ge(self._arange[:size], num_real, out=self._padding[:size])

    def _capture(
        self,
        num_tokens: int,
        inputs: tuple[torch.Tensor | None, ...],
        context: ForwardContext,
    ) -> tuple[torch.Tensor, ...]:
        """Capture the replay layers' graph for ``num_tokens`` rows on the
        persistent input buffers, under the replay batch's forward context."""
        device = next(x for x in inputs if x is not None).device
        cap = max(self.max_num_tokens, num_tokens)
        if self._in_bufs is None:
            self._in_bufs = [
                None
                if x is None
                else torch.zeros((cap, *x.shape[1:]), dtype=x.dtype, device=device)
                for x in inputs
            ]
            self._padding = torch.zeros(cap, dtype=torch.bool, device=device)
            self._arange = torch.arange(cap, device=device)
        if self.graph_pool is None:
            self.graph_pool = torch.cuda.graph_pool_handle()
        for buf, x in zip(self._in_bufs, inputs):
            if x is not None and buf is not None:
                buf[:num_tokens].copy_(x)
        self._set_padding(self.graph_num_tokens, num_tokens)
        views = tuple(
            None if buf is None else buf[:num_tokens] for buf in self._in_bufs
        )

        outer_padding = context.is_padding
        context.is_padding = self._padding[:num_tokens]
        self._inherit_batch_context(context)
        graph = BreakableCUDAGraphCapture(pool=self.graph_pool)
        try:
            with override_forward_context(context), graph:
                outputs = self.run_layers(*views)
                outputs = weak_ref_tensors(outputs)
        finally:
            context.is_padding = outer_padding
        self._graphs[num_tokens] = graph
        self._graph_outputs[num_tokens] = tuple(outputs)
        bisect.insort(self._graph_sizes, num_tokens)

        if self._out_bufs is None:
            self._out_bufs = [
                torch.zeros((cap, *o.shape[1:]), dtype=o.dtype, device=device)
                for o in outputs
            ]
        for out, src in zip(self._out_bufs, outputs):
            out[:num_tokens].copy_(src[:num_tokens])
        return tuple(out[:num_tokens] for out in self._out_bufs)

    # --- entry ---------------------------------------------------------------

    def __call__(
        self, hidden_states: torch.Tensor | MoEOutput, *states: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        if self._seam_applies(hidden_states):
            assert isinstance(hidden_states, torch.Tensor)
            num_tokens = hidden_states.shape[0]
            inputs = tuple(
                None if t is None else weak_ref_tensor(t)
                for t in (hidden_states, *states)
            )
            capture = BreakableCUDAGraphCapture.current()
            assert capture is not None
            return capture.add_eager(lambda: self._seam(num_tokens, inputs))

        rows = self.rows
        if rows is None:
            return self.run_layers(hidden_states, *states)
        # A trimming step holds a prefill longer than the window, more tokens
        # than any step whose MoE leaves its finalize to the next layer.
        assert isinstance(hidden_states, torch.Tensor)
        num_rows = rows.shape[0]
        for buf in self.row_buffers:
            buf[:num_rows].copy_(buf.index_select(0, rows))
        with override_forward_context(self.forward_context):
            row_outputs = self.run_layers(
                hidden_states.index_select(0, rows),
                *(None if t is None else t.index_select(0, rows) for t in states),
            )
        # The trimmed rows' outputs stay zero; nothing reads them.
        num_tokens = hidden_states.shape[0]
        return tuple(
            out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out)
            for out in row_outputs
        )
