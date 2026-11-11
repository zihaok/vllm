# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused kernels for DSpark's sequential Markov sampling.

Per draft position, the eager loop runs an embedding lookup, the Markov bias
GEMM, a logits add, index/copy glue, the Gumbel kernel, an argmax reduction, a
gather and a scatter into ``draft_tokens``. The fused path keeps the bias GEMM
and replaces the rest with two kernels per position.

Each program of ``_markov_sample_kernel`` handles one ``(request, vocab block)``
pair. It forms the logits as ``base_logits[:, i] + bias`` does (fp32 add rounded
to the result dtype) and writes them to the draft-logits cache. For a sampled
request it stores the block's max and sum of ``exp2(logits * log2(e) / temp -
max)``; for a greedy one, the block's max and its first index.
``_markov_select_kernel`` then picks each request's token and copies its Markov
embedding row into the next position's GEMM input.

Sampled tokens are drawn by inverse CDF rather than Gumbel-max: one uniform per
request (keyed by seed, draft step and position) selects a block by the prefix
sum of the block masses and a token by the prefix sum of that block's weights,
the same fp32 values the block mass summed. The tokens follow
softmax(logits / temp) of the cached logits, the distribution the rejection
sampler verifies against, at one exp per logit instead of a hash and two logs.
Greedy tokens are the eager loop's argmax.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    murmur3_uniform32,
    spec_step_seed,
)

_BLOCK_SIZE = 2048
_SAMPLE_NUM_WARPS = 2
# Fused bias + sample: vocab block, K step, pipeline depth and warps.
_FUSED_BLOCK_SIZE = 128
_FUSED_BLOCK_K = 64
_FUSED_NUM_STAGES = 3
_FUSED_NUM_WARPS = 4
# One MMA tile of requests: larger request blocks run slower than the GEMM.
FUSED_MAX_REQS = 32
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _markov_embed_anchor_kernel(
    input_ids_ptr,
    query_per_req,
    weight_ptr,
    weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    BLOCK_R: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    token = tl.load(input_ids_ptr + req * query_per_req).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    row = tl.load(weight_ptr + token * weight_stride + r, mask=r < rank)
    tl.store(embed_ptr + req * embed_stride + r, row, mask=r < rank)


@triton.jit(do_not_specialize=["step"])
def _markov_sample_kernel(
    # [num_reqs, num_steps, vocab_size]
    base_logits_ptr,
    base_stride_0,
    base_stride_1,
    # [num_reqs, vocab_size]
    bias_ptr,
    bias_stride,
    step,
    num_steps,
    # [num_reqs * num_steps], ordered (req, step)
    idx_mapping_ptr,
    temp_ptr,
    # [max_num_reqs, num_steps, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    # [max_num_reqs, BLOCK_P] scratch
    partial_max_ptr,
    partial_idx_ptr,
    partial_sum_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    base = tl.load(
        base_logits_ptr + req * base_stride_0 + step * base_stride_1 + block,
        mask=mask,
        other=float("-inf"),
    )
    bias = tl.load(bias_ptr + req * bias_stride + block, mask=mask, other=0.0)
    # The eager add's rounding: fp32 sum, rounded once to the cache dtype.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(
        logits_cache_ptr.dtype.element_ty
    )

    row = req * num_steps + step
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    tl.store(
        logits_cache_ptr
        + req_state_idx * logits_cache_stride_0
        + step * logits_cache_stride_1
        + block,
        logits,
        mask=mask & is_valid_req,
    )
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    logits = tl.where(mask, logits.to(tl.float32), float("-inf"))
    out = req * BLOCK_P + block_idx
    if temp != 0.0:
        scaled = logits * (_LOG2E / temp)
        block_max = tl.max(scaled, axis=0)
        weights = tl.where(mask, tl.exp2(scaled - block_max), 0.0)
        tl.store(partial_max_ptr + out, block_max)
        tl.store(partial_sum_ptr + out, tl.sum(weights, axis=0))
    else:
        value, idx = tl.max(logits, axis=0, return_indices=True)
        tl.store(partial_max_ptr + out, value)
        tl.store(partial_idx_ptr + out, block_idx * BLOCK_SIZE + idx)


@triton.jit(do_not_specialize=["step", "num_reqs"])
def _markov_bias_sample_kernel(
    # [num_reqs, rank]
    embed_ptr,
    embed_stride,
    # [vocab_size, rank]: the Markov bias projection
    bias_weight_ptr,
    bias_weight_stride,
    # [num_reqs, num_steps, vocab_size]
    base_logits_ptr,
    base_stride_0,
    base_stride_1,
    step,
    num_steps,
    num_reqs,
    # [num_reqs * num_steps], ordered (req, step)
    idx_mapping_ptr,
    temp_ptr,
    # [max_num_reqs, num_steps, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    # [max_num_reqs, BLOCK_P] scratch
    partial_max_ptr,
    partial_idx_ptr,
    partial_sum_ptr,
    vocab_size,
    rank,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_REQ: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """``_markov_sample_kernel`` with the bias GEMM folded in: each program
    computes the bias of one vocab block for a block of requests, so the
    projection is read once per request block."""
    block_idx = tl.program_id(0)
    reqs = tl.program_id(1) * BLOCK_REQ + tl.arange(0, BLOCK_REQ)
    req_mask = reqs < num_reqs
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    # The bias GEMM's fp32 accumulation, rounded to its output dtype.
    acc = tl.zeros((BLOCK_REQ, BLOCK_SIZE), dtype=tl.float32)
    for k in tl.range(0, rank, BLOCK_K, num_stages=NUM_STAGES):
        r = k + tl.arange(0, BLOCK_K)
        embed = tl.load(
            embed_ptr + reqs[:, None] * embed_stride + r[None, :],
            mask=req_mask[:, None] & (r[None, :] < rank),
            other=0.0,
        )
        weight = tl.load(
            bias_weight_ptr
            + block[:, None].to(tl.int64) * bias_weight_stride
            + r[None, :],
            mask=mask[:, None] & (r[None, :] < rank),
            other=0.0,
        )
        acc = tl.dot(embed, tl.trans(weight), acc)
    bias = acc.to(bias_weight_ptr.dtype.element_ty)

    reqs64 = reqs.to(tl.int64)
    tile_mask = req_mask[:, None] & mask[None, :]
    base = tl.load(
        base_logits_ptr
        + reqs64[:, None] * base_stride_0
        + step * base_stride_1
        + block[None, :],
        mask=tile_mask,
        other=float("-inf"),
    )
    # The eager add's rounding: fp32 sum, rounded once to the cache dtype.
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(
        logits_cache_ptr.dtype.element_ty
    )

    req_state_idx = tl.load(
        idx_mapping_ptr + reqs64 * num_steps + step, mask=req_mask, other=-1
    ).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    tl.store(
        logits_cache_ptr
        + req_state_idx[:, None] * logits_cache_stride_0
        + step * logits_cache_stride_1
        + block[None, :],
        logits,
        mask=tile_mask & is_valid_req[:, None],
    )
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    logits = tl.where(mask[None, :], logits.to(tl.float32), float("-inf"))
    out = reqs64 * BLOCK_P + block_idx
    sampled = temp != 0.0
    scaled = logits * (_LOG2E / tl.where(sampled, temp, 1.0))[:, None]
    block_max = tl.max(scaled, axis=1)
    weights = tl.where(mask[None, :], tl.exp2(scaled - block_max[:, None]), 0.0)
    value, idx = tl.max(logits, axis=1, return_indices=True)
    tl.store(partial_max_ptr + out, tl.where(sampled, block_max, value), mask=req_mask)
    tl.store(partial_sum_ptr + out, tl.sum(weights, axis=1), mask=req_mask & sampled)
    tl.store(
        partial_idx_ptr + out,
        block_idx * BLOCK_SIZE + idx,
        mask=req_mask & (temp == 0.0),
    )


@triton.jit(do_not_specialize=["step"])
def _markov_select_kernel(
    step,
    num_steps,
    idx_mapping_ptr,
    sample_pos_ptr,
    temp_ptr,
    seeds_ptr,
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    draft_tokens_ptr,
    draft_tokens_stride,
    embed_weight_ptr,
    embed_weight_stride,
    embed_ptr,
    embed_stride,
    rank,
    partial_max_ptr,
    partial_idx_ptr,
    partial_sum_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_NEXT: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    row = req * num_steps + step
    req_state_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    p = tl.arange(0, BLOCK_P)
    p_mask = p < num_blocks
    values = tl.load(
        partial_max_ptr + req * BLOCK_P + p, mask=p_mask, other=float("-inf")
    )
    if temp != 0.0:
        seed = spec_step_seed(
            tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0), step
        )
        # The draw is keyed by the position before the predicted token.
        pos = tl.load(sample_pos_ptr + row) - 1 + _DRAFT_NOISE_SALT
        sums = tl.load(partial_sum_ptr + req * BLOCK_P + p, mask=p_mask, other=0.0)
        top = tl.max(values, axis=0)
        mass = tl.where(sums > 0, sums * tl.exp2(values - top), 0.0)
        cum = tl.cumsum(mass, axis=0)
        target = murmur3_uniform32(seed, pos, vocab_size) * tl.sum(mass, axis=0)
        # A rounding overshoot past the last block takes the last nonzero one.
        last = tl.max(tl.where(mass > 0, p, 0), axis=0)
        blk = tl.minimum(tl.sum((p_mask & (cum <= target)).to(tl.int32)), last)
        at_blk = p == blk
        blk_max = tl.sum(tl.where(at_blk, values, 0.0))
        # The remaining mass, in units of the block's own weights.
        target = (target - tl.sum(tl.where(at_blk, cum - mass, 0.0))) * tl.exp2(
            top - blk_max
        )

        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        cached = tl.load(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + step * logits_cache_stride_1
            + offs,
            mask=mask & is_valid_req,
            other=float("-inf"),
        )
        # The same fp32 weights _markov_sample_kernel summed for this block.
        scaled = cached.to(tl.float32) * (_LOG2E / temp)
        weights = tl.where(mask, tl.exp2(scaled - blk_max), 0.0)
        i = tl.arange(0, BLOCK_SIZE)
        last_tok = tl.max(tl.where(weights > 0, i, 0), axis=0)
        cum_tok = tl.cumsum(weights, axis=0)
        idx = tl.minimum(tl.sum((cum_tok <= target).to(tl.int32)), last_tok)
        token = (blk * BLOCK_SIZE + idx).to(tl.int64)
    else:
        # torch.argmax semantics: the first NaN if any, else the first maximum.
        is_nan = values != values
        first_nan = tl.min(tl.where(is_nan, p, BLOCK_P))
        best = tl.max(tl.where(is_nan, float("-inf"), values))
        first_max = tl.min(tl.where(p_mask & (values == best), p, BLOCK_P))
        winner = tl.where(first_nan < BLOCK_P, first_nan, first_max)
        token = tl.load(partial_idx_ptr + req * BLOCK_P + winner).to(tl.int64)
    tl.store(draft_tokens_ptr + req * draft_tokens_stride + step, token)
    if HAS_NEXT:
        r = tl.arange(0, BLOCK_R)
        embed_row = tl.load(
            embed_weight_ptr + token * embed_weight_stride + r, mask=r < rank
        )
        tl.store(embed_ptr + req * embed_stride + r, embed_row, mask=r < rank)


class DSparkFusedMarkovSampler:
    """Persistent buffers and launchers of the fused Markov sampling loop.

    ``embed_weight`` is the ``[vocab, rank]`` table ``markov_embed`` looks up.
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        embed_weight: torch.Tensor,
    ):
        assert embed_weight.dim() == 2 and embed_weight.stride(1) == 1
        device = embed_weight.device
        self.embed_weight = embed_weight
        self.vocab_size = vocab_size
        self.num_blocks = triton.cdiv(vocab_size, _BLOCK_SIZE)
        self.block_p = triton.next_power_of_2(self.num_blocks)
        self.rank = embed_weight.shape[1]
        self.block_r = triton.next_power_of_2(self.rank)
        # Input of the Markov bias GEMM; rewritten by every sampling step.
        self.markov_embed = torch.zeros(
            max_num_reqs, self.rank, dtype=embed_weight.dtype, device=device
        )
        # The fused bias + sample kernel uses smaller vocab blocks, so that its
        # programs together read the bias projection at full bandwidth.
        self.fused_num_blocks = triton.cdiv(vocab_size, _FUSED_BLOCK_SIZE)
        self.fused_block_p = triton.next_power_of_2(self.fused_num_blocks)
        shape = (max_num_reqs, max(self.block_p, self.fused_block_p))
        self.partial_max = torch.empty(shape, dtype=torch.float32, device=device)
        self.partial_idx = torch.empty(shape, dtype=torch.int32, device=device)
        self.partial_sum = torch.empty(shape, dtype=torch.float32, device=device)

    def embed_anchor(
        self, input_ids: torch.Tensor, query_per_req: int, num_reqs: int
    ) -> torch.Tensor:
        """Markov embedding of each request's anchor ``input_ids[r * q]``."""
        embed = self.markov_embed[:num_reqs]
        _markov_embed_anchor_kernel[(num_reqs,)](
            input_ids,
            query_per_req,
            self.embed_weight,
            self.embed_weight.stride(0),
            embed,
            embed.stride(0),
            self.rank,
            BLOCK_R=self.block_r,
        )
        return embed

    def sample_step(
        self,
        base_logits: torch.Tensor,  # [num_reqs, num_steps, vocab_size]
        bias: torch.Tensor,  # [num_reqs, vocab_size], or [vocab_size, rank] fused
        step: int,
        idx_mapping: torch.Tensor,  # [num_reqs * num_steps]
        sample_pos: torch.Tensor,  # [num_reqs * num_steps]
        temperature: torch.Tensor,  # [max_num_reqs]
        seeds: torch.Tensor,  # [max_num_reqs]
        logits_cache: torch.Tensor,  # [max_num_reqs, num_steps, vocab_size]
        draft_tokens: torch.Tensor,  # [max_num_reqs, num_steps]
        fuse_bias: bool = False,
    ) -> None:
        """Sample position ``step`` into ``draft_tokens[:, step]`` and, unless
        it is the last position, leave the sampled tokens' Markov embeddings in
        ``self.markov_embed`` for the next bias GEMM. With ``fuse_bias``,
        ``bias`` is the Markov projection and the bias GEMM runs inside the
        sampling kernel (at most ``FUSED_MAX_REQS`` requests)."""
        num_reqs, num_steps, vocab_size = base_logits.shape
        assert vocab_size == self.vocab_size
        assert base_logits.stride(2) == 1 and bias.stride(-1) == 1
        assert logits_cache.size(-1) >= vocab_size and logits_cache.stride(2) == 1
        # The eager path caches base_logits + bias in the cache's dtype.
        assert torch.promote_types(base_logits.dtype, bias.dtype) == (
            logits_cache.dtype
        )
        embed = self.markov_embed[:num_reqs]
        if fuse_bias:
            assert num_reqs <= FUSED_MAX_REQS and bias.size(0) >= vocab_size
            block_size, num_blocks = _FUSED_BLOCK_SIZE, self.fused_num_blocks
            block_p = self.fused_block_p
            block_req = max(16, triton.next_power_of_2(num_reqs))
            _markov_bias_sample_kernel[(num_blocks, triton.cdiv(num_reqs, block_req))](
                embed,
                embed.stride(0),
                bias,
                bias.stride(0),
                base_logits,
                base_logits.stride(0),
                base_logits.stride(1),
                step,
                num_steps,
                num_reqs,
                idx_mapping,
                temperature,
                logits_cache,
                logits_cache.stride(0),
                logits_cache.stride(1),
                self.partial_max,
                self.partial_idx,
                self.partial_sum,
                vocab_size,
                self.rank,
                BLOCK_SIZE=block_size,
                BLOCK_REQ=block_req,
                BLOCK_K=_FUSED_BLOCK_K,
                NUM_STAGES=_FUSED_NUM_STAGES,
                BLOCK_P=block_p,
                num_warps=_FUSED_NUM_WARPS,
            )
        else:
            block_size, num_blocks, block_p = _BLOCK_SIZE, self.num_blocks, self.block_p
            _markov_sample_kernel[(num_reqs, num_blocks)](
                base_logits,
                base_logits.stride(0),
                base_logits.stride(1),
                bias,
                bias.stride(0),
                step,
                num_steps,
                idx_mapping,
                temperature,
                logits_cache,
                logits_cache.stride(0),
                logits_cache.stride(1),
                self.partial_max,
                self.partial_idx,
                self.partial_sum,
                vocab_size,
                BLOCK_SIZE=block_size,
                BLOCK_P=block_p,
                num_warps=_SAMPLE_NUM_WARPS,
            )
        _markov_select_kernel[(num_reqs,)](
            step,
            num_steps,
            idx_mapping,
            sample_pos,
            temperature,
            seeds,
            logits_cache,
            logits_cache.stride(0),
            logits_cache.stride(1),
            draft_tokens,
            draft_tokens.stride(0),
            self.embed_weight,
            self.embed_weight.stride(0),
            embed,
            embed.stride(0),
            self.rank,
            self.partial_max,
            self.partial_idx,
            self.partial_sum,
            vocab_size,
            num_blocks,
            BLOCK_SIZE=block_size,
            BLOCK_P=block_p,
            BLOCK_R=self.block_r,
            HAS_NEXT=step + 1 < num_steps,
        )
