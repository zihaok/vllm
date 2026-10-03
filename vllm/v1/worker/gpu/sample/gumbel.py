# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used by the
# retained Philox helper for rejection sampling.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10)

# Offset salt keeping the draft's Gumbel noise disjoint from the target's.
# Verification is a probability-ratio test, not a Gumbel coupling, so a proposal
# and the residual it is resampled from must not share a noise vector.
# Positions are int64 and never approach 2**30, so the streams cannot collide.
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def _murmur3_rotl32(value, shift: tl.constexpr):
    return (value << shift) | (value >> (32 - shift))


@triton.jit
def _murmur3_mix(h, key):
    key *= 0xCC9E2D51
    key = _murmur3_rotl32(key, 15)
    key *= 0x1B873593
    h ^= key
    h = _murmur3_rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_fmix32(h):
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    return h ^ (h >> 16)


@triton.jit
def murmur3_hash32(seed, pos, offset, domain: tl.constexpr = 0):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    # Keep the request-wide prefix scalar until the token offset is mixed in.
    h = (seed ^ seed).to(tl.uint32)
    h ^= domain
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset)
    return _murmur3_fmix32(h ^ 16)


@triton.jit
def murmur3_uniform32(seed, pos, offset):
    return _uniform32_from_random(murmur3_hash32(seed, pos, offset))


@triton.jit
def _uniform32_from_random(random32):
    # Split the uint32 before converting to fp32 so backends without a native
    # uint32-to-float conversion can still use all 32 source bits. Both 16-bit
    # halves convert exactly; their sum is the correctly rounded fp32 value of
    # (random32 + 0.5) * 2**-32. In particular, the u -> 0 winning tail keeps
    # the full 32-bit source resolution instead of being truncated to 24 bits.
    hi16 = (random32 >> 16).to(tl.int32)
    lo16 = (random32 & 0xFFFF).to(tl.int32)
    return (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def _uniform64_from_random53(random53):
    uniform = (random53.to(tl.float64) + 0.5) * 1.1102230246251565e-16
    # The largest midpoint rounds to 1.0 in fp64; keep the uniform open without
    # relying on a near-one literal that Triton may materialize in fp32.
    return tl.where(uniform == 1.0, uniform - 1.1102230246251565e-16, uniform)


@triton.jit
def murmur3_uniform64(seed, pos, offset):
    lo = murmur3_hash32(seed, pos, offset).to(tl.uint64)
    hi = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9).to(tl.uint64)
    random53 = ((hi << 32) | lo) >> 11
    return _uniform64_from_random53(random53)


@triton.jit
def _log1p_neg_stable(value):
    # Preserve precision for the positive Gumbel tail without relying on a
    # backend-specific libdevice log1p. The degree-8 series has absolute error
    # below 6e-7 on [0, 0.25]; subtraction is well-conditioned elsewhere for
    # the part of the distribution that can win the argmax.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial

    direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)


@triton.jit
def gumbel_noise32(u):
    """fp32 Gumbel noise of gumbel_noised_argmax for a murmur3_uniform32 draw."""
    return -tl.log(-_log1p_neg_stable(u))


# Max |_approx_gumbel_noise32(u) - gumbel_noise32(u)| over every u that
# _uniform32_from_random can return (all 2**32 inputs), with margin. The
# exhaustive check in kernels/sampler/bench/test_gumbel.py (--suite bound)
# measures the true maximum on the target GPU and requires it to be at most a
# quarter of this.
_APPROX_GUMBEL_ERR = tl.constexpr(2e-4)


@triton.jit
def _approx_gumbel_noise32(u):
    """Cheap approximation of gumbel_noise32(u), |error| <= _APPROX_GUMBEL_ERR.

    gumbel_noise32 = -ln(L) with L = -ln(1 - u) costs two libdevice logf.
    Here w = L / ln2 = -log2(1 - u) comes from (u + u^2/2 + u^3/3) / ln2 for
    small u and from the MUFU lg2 of the same clamped 1 - u otherwise, and
    -ln(L) = -ln2 * log2(w) - ln(ln2) takes a second MUFU lg2. It is only
    used to screen out tokens that cannot win the Gumbel argmax; the
    candidates are recomputed with gumbel_noise32.
    """
    w_small = u * (
        1.4426950408889634 + u * (0.7213475204444817 + u * 0.4808983469629878)
    )
    w_large = -tldevice.fast_log2f(tl.maximum(1.0 - u, 5.960464477539063e-08))
    w = tl.where(u < 0.015625, w_small, w_large)
    return tldevice.fast_log2f(w) * -0.6931471805599453 + 0.36651292058166435


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        if USE_FP64:
            u = murmur3_uniform64(seed, pos, keys)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = murmur3_uniform32(seed, pos, keys)
            # Draw the large-noise tail (which decides the argmax winner) from
            # u -> 0, where fp32 has fine resolution. Avoid backend-specific
            # log1p while preserving precision in the winning tail.
            gumbel_noise = -tl.log(-_log1p_neg_stable(u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_source_ptr,
    logits_cache_source_stride,
    vocab_size,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if logits_cache_ptr is not None:
        # Store the logits *before* temperature. Dividing first would produce a
        # value that is generally not representable in the cache's dtype, forcing
        # it to be fp32. Consumers (the rejection sampler) divide by the same
        # temperature on load, which reproduces the value used below bitwise.
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        cached_logits = tl.load(
            logits_cache_source_ptr + token_idx * logits_cache_source_stride + block,
            mask=mask,
        )
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
            + block,
            cached_logits,
            mask=mask & is_valid_req,
        )

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    return gumbel_noised_argmax(
        logits,
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_source_ptr,
    logits_cache_source_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        logits_cache_source_ptr,
        logits_cache_source_stride,
        vocab_size,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


@triton.jit
def _screened_exact_noised_logit(
    row_ptr,
    bias_row_ptr,
    token_id,
    mask,
    seed,
    pos,
    temp,
    HAS_BIAS: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    """gumbel_noised_argmax's noised logit at token_id, recomputed exactly.

    Same operations as the reference kernel's vector code, on any shape of
    token ids (masked-off entries return -inf).
    """
    raw = tl.load(row_ptr + token_id, mask=mask, other=float("-inf"))
    if HAS_BIAS:
        raw = (
            raw.to(tl.float32)
            + tl.load(bias_row_ptr + token_id, mask=mask, other=0.0).to(tl.float32)
        ).to(row_ptr.dtype.element_ty)
    x = raw.to(tl.float32)
    if APPLY_TEMPERATURE:
        x = x / temp
    u = _uniform32_from_random(murmur3_hash32(seed, pos, token_id))
    return tl.where(mask, x + gumbel_noise32(u), float("-inf"))


@triton.jit
def _gumbel_screen_kernel(
    # [num_tokens, num_blocks]
    local_value_ptr,
    # [num_tokens, num_blocks]: token id of the block max
    local_token_ptr,
    # [num_tokens, num_blocks]: 1 if local_value still needs the exact value
    # of local_token (one candidate), 0 if local_value is already exact.
    local_pending_ptr,
    local_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    bias_ptr,
    bias_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
):
    """Per-block (max, lowest argmax) of the fp32 Gumbel-noised logits, the
    values _gumbel_sample_kernel produces, at a fraction of the arithmetic.

    The exact noise costs two libdevice logf per token. Here every token gets
    the cheap _approx_gumbel_noise32, within _APPROX_GUMBEL_ERR of the exact
    noise, which bounds every exact noised logit to within a tolerance of its
    approximation. Only tokens whose approximation comes within twice that
    tolerance of the block's approximate max can hold the exact max. Almost
    always that is a single token; the block then records it as pending and
    _gumbel_finalize_kernel computes its exact value. With a few candidates
    the block evaluates them exactly here, and with many (or a non-finite
    max) it falls back to exact noise for every token. Either way the
    resulting block max and its lowest token index are exact.

    With HAS_BIAS the sampled (and cached) logits are logits + bias, rounded
    to the logits dtype like the PyTorch add they replace.
    """
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    row_ptr = logits_ptr + token_idx * logits_stride
    bias_row_ptr = bias_ptr + token_idx * bias_stride
    raw = tl.load(row_ptr + block, mask=mask, other=float("-inf"))
    if HAS_BIAS:
        raw = (
            raw.to(tl.float32)
            + tl.load(bias_row_ptr + block, mask=mask, other=0.0).to(tl.float32)
        ).to(logits_ptr.dtype.element_ty)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if logits_cache_ptr is not None:
        # Pre-temperature logits; see gumbel_block_argmax.
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
            + block,
            raw,
            mask=mask & is_valid_req,
        )

    x = raw.to(tl.float32)
    pending = 0
    if temp == 0.0:
        value, idx = tl.max(x, axis=0, return_indices=True)
    else:
        if APPLY_TEMPERATURE:
            x = x / temp
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        u = _uniform32_from_random(murmur3_hash32(seed, pos, block))
        # Masked lanes hold -inf logits, so their approximation is -inf too.
        approx = x + _approx_gumbel_noise32(u)
        approx_max, approx_idx = tl.max(approx, axis=0, return_indices=True)
        # |exact - approx| <= E + fp32 rounding of the add, per token, so any
        # token whose exact value can reach the exact max lies within twice
        # that of the approximate max.
        tol = 2.5 * _APPROX_GUMBEL_ERR + tl.abs(approx_max) * 9.5367431640625e-07
        # `not <` also selects NaN, which forces the exact fallback below.
        candidates = ~(approx < approx_max - tol)
        num_candidates = tl.sum(candidates.to(tl.int32))
        finite = tl.abs(approx_max) < 3.0e38
        if approx_max == float("-inf"):
            # Every logit is -inf (e.g. a top-k draft): so is every noised
            # logit, and tl.max picks index 0.
            value = -float("inf")
            idx = 0
        elif (num_candidates == 1) & finite:
            value = approx_max
            idx = approx_idx
            pending = 1
        elif (num_candidates <= MAX_CANDIDATES) & finite:
            # Candidates in increasing token order; strict `>` keeps the
            # lowest index among equal maxima, like tl.max.
            value = -float("inf")
            idx = 0
            prev = -1
            for _ in range(num_candidates):
                token_id = tl.min(
                    tl.where(candidates & (block > prev), block, 2147483647)
                )
                v = _screened_exact_noised_logit(
                    row_ptr,
                    bias_row_ptr,
                    token_id,
                    True,
                    seed,
                    pos,
                    temp,
                    HAS_BIAS=HAS_BIAS,
                    APPLY_TEMPERATURE=APPLY_TEMPERATURE,
                )
                if v > value:
                    value = v
                    idx = token_id - block_idx * BLOCK_SIZE
                prev = token_id
        else:
            exact = tl.where(mask, x + gumbel_noise32(u), float("-inf"))
            value, idx = tl.max(exact, axis=0, return_indices=True)
    out = token_idx * local_stride + block_idx
    tl.store(local_value_ptr + out, value)
    tl.store(local_token_ptr + out, block_idx * BLOCK_SIZE + idx)
    tl.store(local_pending_ptr + out, pending)


@triton.jit
def _gumbel_finalize_kernel(
    sampled_ptr,
    # Optional [num_tokens] fp32: the row's max noised logit (for tests).
    sampled_value_ptr,
    local_value_ptr,
    local_token_ptr,
    local_pending_ptr,
    local_stride,
    num_blocks,
    logits_ptr,
    logits_stride,
    bias_ptr,
    bias_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    """Exact values of the pending blocks, then the row argmax (lowest block
    on ties, like torch.argmax over the reference kernel's block maxima)."""
    token_idx = tl.program_id(0).to(tl.int64)
    blocks = tl.arange(0, PADDED_NUM_BLOCKS)
    bmask = blocks < num_blocks
    base = token_idx * local_stride
    value = tl.load(local_value_ptr + base + blocks, mask=bmask, other=-float("inf"))
    token = tl.load(local_token_ptr + base + blocks, mask=bmask, other=0)
    pending = tl.load(local_pending_ptr + base + blocks, mask=bmask, other=0) != 0
    if tl.max(pending.to(tl.int32)) > 0:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
            tl.float32
        )
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        exact = _screened_exact_noised_logit(
            logits_ptr + token_idx * logits_stride,
            bias_ptr + token_idx * bias_stride,
            token,
            pending,
            seed,
            pos,
            temp,
            HAS_BIAS=HAS_BIAS,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )
        value = tl.where(pending, exact, value)
    best = tl.argmax(value, axis=0)
    tl.store(sampled_ptr + token_idx, tl.sum(tl.where(blocks == best, token, 0)))
    if sampled_value_ptr is not None:
        tl.store(sampled_value_ptr + token_idx, tl.max(value, axis=0))


_SCREENED_BLOCK_SIZE = 1024
_SCREENED_NUM_WARPS = 4


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
    logits_cache_source: torch.Tensor | None = None,
    logits_bias: torch.Tensor | None = None,  # [num_tokens, vocab_size]
) -> torch.Tensor:
    """Gumbel-max sample of (logits [+ logits_bias]) / temperature per row.

    `logits_bias`, if given, is added to `logits` (rounded to the logits
    dtype, as `logits + logits_bias` would be) inside the kernel; the sum is
    what gets sampled and cached.
    """
    if logits_bias is not None and not _use_screened_kernel(
        logits, logits_bias, use_fp64, logits_cache_source
    ):
        logits = logits + logits_bias
        logits_bias = None
    if _use_screened_kernel(logits, logits_bias, use_fp64, logits_cache_source):
        return _gumbel_sample_screened(
            logits,
            logits_bias,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            is_drafting,
            logits_cache,
            logits_cache_col,
        )
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        if logits_cache_source is None:
            logits_cache_source = logits
        assert logits_cache_source.shape == logits.shape, (
            "logits cache source must match sampled logits shape"
        )
        assert logits_cache_source.device == logits.device, (
            "logits cache source must be on the sampled logits device"
        )
        assert logits_cache_source.dtype == logits_cache.dtype, (
            "logits cache source and destination must have the same dtype"
        )
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
    elif logits_cache_source is not None:
        raise ValueError("logits_cache_source requires logits_cache")
    if logits_cache_source is not None and logits_cache_source.stride(-1) != 1:
        logits_cache_source = logits_cache_source.contiguous()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits_cache_source,
        logits_cache_source.stride(0) if logits_cache_source is not None else 0,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


def _use_screened_kernel(
    logits: torch.Tensor,
    logits_bias: torch.Tensor | None,
    use_fp64: bool,
    logits_cache_source: torch.Tensor | None,
) -> bool:
    # The screen relies on the CUDA libdevice fast log2; fp64 noise and a
    # separate cache source keep the reference kernel.
    if use_fp64 or logits_cache_source is not None:
        return False
    if not current_platform.is_cuda() or logits.stride(-1) != 1:
        return False
    if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    return logits_bias is None or (
        logits_bias.shape == logits.shape
        and logits_bias.dtype == logits.dtype
        and logits_bias.stride(-1) == 1
    )


def _gumbel_sample_screened(
    logits: torch.Tensor,
    logits_bias: torch.Tensor | None,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None,
    logits_cache_col: torch.Tensor | None,
    block_size: int = _SCREENED_BLOCK_SIZE,
    num_warps: int = _SCREENED_NUM_WARPS,
    sampled_value: torch.Tensor | None = None,
) -> torch.Tensor:
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.dtype == logits.dtype, (
            "logits cache source and destination must have the same dtype"
        )
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )
    bias = logits_bias if logits_bias is not None else logits
    bias_stride = logits_bias.stride(0) if logits_bias is not None else 0
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_value = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    local_token = logits.new_empty(num_tokens, num_blocks, dtype=torch.int32)
    local_pending = logits.new_empty(num_tokens, num_blocks, dtype=torch.int8)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_screen_kernel[(num_tokens, num_blocks)](
        local_value,
        local_token,
        local_pending,
        local_value.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        bias,
        bias_stride,
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=block_size,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        PER_TOKEN_COL=per_token_col,
        HAS_BIAS=logits_bias is not None,
        MAX_CANDIDATES=16,
        num_warps=num_warps,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    sampled = logits.new_empty(num_tokens, dtype=torch.int64)
    _gumbel_finalize_kernel[(num_tokens,)](
        sampled,
        sampled_value,
        local_value,
        local_token,
        local_pending,
        local_value.stride(0),
        num_blocks,
        logits,
        logits.stride(0),
        bias,
        bias_stride,
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        HAS_BIAS=logits_bias is not None,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    return sampled
