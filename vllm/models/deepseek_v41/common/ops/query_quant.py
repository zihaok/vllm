# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def mxfp8_scale_offset(row, group, GROUPS: tl.constexpr):
    """F8_128x4: [row/128, group/4, row%32, row%128/32, group%4]."""
    offset = (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
    return offset + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


def mxfp8_scale_bytes(tokens: int, width: int) -> int:
    """Bytes of F8_128x4 scales for `tokens` rows of `width` MXFP8 values."""
    return triton.cdiv(tokens, 128) * 128 * triton.cdiv(width // 32, 4) * 4


@triton.jit
def _mxfp8_quant_rows(y, GROUPS: tl.constexpr):
    """[R, GROUPS * 32] FP32 -> (scaled values, UE8M0 exponents [R, GROUPS]),
    rounded as FlashInfer quantizes MXFP8."""
    R: tl.constexpr = y.shape[0]
    grouped = tl.reshape(y, (R, GROUPS, 32))
    amax = tl.max(tl.abs(grouped), 2)
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.minimum(exponent + bump, 254)
    sf = tl.where(normalized <= 0, 0, sf)
    # Match FlashInfer's UE8M0 conversion, including zero/subnormal scales.
    inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
    inv_scale = inv_bits.to(tl.float32, bitcast=True)
    return tl.reshape(grouped * inv_scale[:, :, None], (R, GROUPS * 32)), sf


class Mxfp8KVInsert(NamedTuple):
    """A packed MXFP8 SWA cache insert for ``fused_q_kv_rmsnorm_quant``.

    Records are 512 E4M3 dims then 16 UE8M0 scales, written byte for byte as
    ``torch.ops._C.fused_deepseek_v4_kv_rope_insert(..., kv_mxfp8=True)``
    does: GPT-J RoPE of the last 64 dims, then MXFP8 of the row.
    """

    # [blocks, block bytes] uint8.
    cache: torch.Tensor
    # [<= tokens] int64, -1 for none; tokens past it insert nothing.
    slot_mapping: torch.Tensor
    # [tokens] int64.
    positions: torch.Tensor
    # [max_position, 64] FP32: each position's 32 cos, then its 32 sin.
    cos_sin_cache: torch.Tensor
    block_size: int


class KVCacheInsert(NamedTuple):
    """A per-tensor FP8 paged cache that ``fused_q_kv_rmsnorm_quant`` also
    writes the normalized KV into, byte for byte as
    ``torch.ops._C.fused_deepseek_v4_kv_rope_insert`` writes it: the GPT-J RoPE
    of the last 64 dims, then E4M3 of the KV over ``scale``."""

    # [blocks, block_size, KV width] float8_e4m3fn.
    cache: torch.Tensor
    # [<= tokens] int64, -1 for none; the tokens past it insert nothing.
    slot_mapping: torch.Tensor
    # [tokens] int64.
    positions: torch.Tensor
    # [max_position, 64] FP32: each position's 32 cos, then its 32 sin.
    cos_sin_cache: torch.Tensor
    block_size: int
    # [1] FP32.
    scale: torch.Tensor


@triton.jit
def _gptj_rope(even, odd, cos, sin):
    """``even * cos - odd * sin`` and ``even * sin + odd * cos`` contracted as
    nvcc contracts them in the CUDA insert kernels:
    fma(even, cos, -(odd * sin)) and fma(even, sin, odd * cos)."""
    return tl.inline_asm_elementwise(
        "{\n.reg .f32 t;\n"
        "mul.rn.f32 t, $3, $5;\nneg.f32 t, t;\nfma.rn.f32 $0, $2, $4, t;\n"
        "mul.rn.f32 t, $3, $4;\nfma.rn.f32 $1, $2, $5, t;\n}",
        "=f,=f,f,f,f,f",
        [even, odd, cos, sin],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mxfp8_kv_insert(
    y,
    rows,
    cache,
    slots,
    positions,
    cos_sin,
    num_slots,
    block_stride,
    KV_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    """Insert the [ROWS, KV_SIZE] FP32 normalized KV rows ``y`` (rounded to
    BF16 here, as the KV tensor is) into the packed MXFP8 SWA cache."""
    ROWS: tl.constexpr = y.shape[0]
    PAIRS: tl.constexpr = KV_SIZE // 2
    ROPE_PAIRS: tl.constexpr = 32
    insert = rows < num_slots
    slot = tl.load(slots + rows, insert, other=-1)
    position = tl.load(positions + rows, insert, other=0)
    pair = tl.arange(0, PAIRS) - (PAIRS - ROPE_PAIRS)
    rope = (pair >= 0)[None, :] & insert[:, None]
    cs = cos_sin + position[:, None] * (2 * ROPE_PAIRS) + pair[None, :]
    cos = tl.load(cs, rope, other=1.0)
    sin = tl.load(cs + ROPE_PAIRS, rope, other=0.0)
    y = y.to(tl.bfloat16).to(tl.float32)
    even, odd = tl.split(tl.reshape(y, (ROWS, PAIRS, 2)))
    roped_even, roped_odd = _gptj_rope(even, odd, cos, sin)
    even = tl.where(rope, roped_even, even)
    odd = tl.where(rope, roped_odd, odd)
    y = tl.reshape(tl.join(even, odd), (ROWS, KV_SIZE))
    y = y.to(tl.bfloat16).to(tl.float32)
    GROUPS: tl.constexpr = KV_SIZE // 32
    grouped = tl.reshape(y, (ROWS, GROUPS, 32))
    absmax = tl.maximum(tl.max(tl.abs(grouped), 2), 1e-4)
    exponent = tl.math.ceil(tl.extra.cuda.libdevice.log2(tl.math.div_rn(absmax, 448.0)))
    inv_scale = tl.extra.cuda.libdevice.exp2(-exponent)
    scaled = tl.minimum(tl.maximum(grouped * inv_scale[:, :, None], -448.0), 448.0)
    data = (
        tl.reshape(scaled, (ROWS, KV_SIZE)).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    )
    sf = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    block = slot // CACHE_BLOCK
    record = (
        cache + block * block_stride + (slot - block * CACHE_BLOCK) * (KV_SIZE + GROUPS)
    )
    live = slot >= 0
    c = tl.arange(0, KV_SIZE)
    tl.store(record[:, None] + c[None, :], data, live[:, None])
    g = tl.arange(0, GROUPS)
    tl.store(record[:, None] + KV_SIZE + g[None, :], sf, live[:, None])


@triton.jit
def _fp8_kv_insert(
    y,
    rows,
    cache,
    slots,
    positions,
    cos_sin,
    kv_scale,
    num_slots,
    block_stride,
    token_stride,
    KV_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    """Insert the [ROWS, KV_SIZE] FP32 normalized KV rows ``y`` (rounded to
    BF16 here, as the KV tensor is) into the per-tensor FP8 paged cache."""
    ROWS: tl.constexpr = y.shape[0]
    PAIRS: tl.constexpr = KV_SIZE // 2
    ROPE_PAIRS: tl.constexpr = 32
    insert = rows < num_slots
    slot = tl.load(slots + rows, insert, other=-1)
    position = tl.load(positions + rows, insert, other=0)
    pair = tl.arange(0, PAIRS) - (PAIRS - ROPE_PAIRS)
    rope = (pair >= 0)[None, :] & insert[:, None]
    cs = cos_sin + position[:, None] * (2 * ROPE_PAIRS) + pair[None, :]
    cos = tl.load(cs, rope, other=1.0)
    sin = tl.load(cs + ROPE_PAIRS, rope, other=0.0)
    y = y.to(tl.bfloat16).to(tl.float32)
    even, odd = tl.split(tl.reshape(y, (ROWS, PAIRS, 2)))
    roped_even, roped_odd = _gptj_rope(even, odd, cos, sin)
    even = tl.where(rope, roped_even, even)
    odd = tl.where(rope, roped_odd, odd)
    y = tl.reshape(tl.join(even, odd), (ROWS, KV_SIZE))
    inv_scale = tl.math.div_rn(1.0, tl.load(kv_scale))
    y = tl.minimum(tl.maximum(y * inv_scale, -448.0), 448.0)
    block = slot // CACHE_BLOCK
    record = cache + block * block_stride + (slot - block * CACHE_BLOCK) * token_stride
    c = tl.arange(0, KV_SIZE)
    tl.store(record[:, None] + c[None, :], y, (slot >= 0)[:, None])


@triton.jit
def _q_norm_quant_rows(
    q,
    qw,
    qo,
    scales,
    rows,
    live,
    q_stride,
    eps,
    Q_SIZE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_TAIL: tl.constexpr,
    Q_TAIL_PAD: tl.constexpr,
    ROWS: tl.constexpr,
):
    # A tail that is not a power of two pads to Q_TAIL_PAD under a mask.
    c_head = tl.arange(0, Q_HEAD)
    c_tail = tl.arange(0, Q_TAIL_PAD)
    g_tail = tl.arange(0, Q_TAIL_PAD // 32)
    if Q_TAIL_PAD == Q_TAIL:
        tail_live = live[:, None]
        w_tail = tl.load(qw + Q_HEAD + c_tail).to(tl.float32)
        sf_tail_mask = None
    else:
        tail_live = live[:, None] & (c_tail < Q_TAIL)[None, :]
        w_tail = tl.load(qw + Q_HEAD + c_tail, c_tail < Q_TAIL, 0).to(tl.float32)
        sf_tail_mask = (g_tail < Q_TAIL // 32)[None, :]
    q_head = tl.load(
        q + rows[:, None] * q_stride + c_head[None, :], live[:, None], 0
    ).to(tl.float32)
    q_tail = tl.load(
        q + rows[:, None] * q_stride + Q_HEAD + c_tail[None, :], tail_live, 0
    ).to(tl.float32)
    rrms = tl.rsqrt(
        (tl.sum(q_head * q_head, 1) + tl.sum(q_tail * q_tail, 1)) / Q_SIZE + eps
    )
    w_head = tl.load(qw + c_head).to(tl.float32)
    # Preserve the materialized normalization output's rounding boundary.
    y_head = (
        (q_head * rrms[:, None] * w_head[None, :]).to(q.dtype.element_ty).to(tl.float32)
    )
    y_tail = (
        (q_tail * rrms[:, None] * w_tail[None, :]).to(q.dtype.element_ty).to(tl.float32)
    )
    quant_head, sf_head = _mxfp8_quant_rows(y_head, Q_HEAD // 32)
    quant_tail, sf_tail = _mxfp8_quant_rows(y_tail, Q_TAIL_PAD // 32)
    tl.store(qo + rows[:, None] * Q_SIZE + c_head[None, :], quant_head, live[:, None])
    tl.store(
        qo + rows[:, None] * Q_SIZE + Q_HEAD + c_tail[None, :], quant_tail, tail_live
    )
    # Rows past num_tokens up to the 128-row scale tile get zero scales.
    g_head = tl.arange(0, Q_HEAD // 32)
    tl.store(
        scales + mxfp8_scale_offset(rows[:, None], g_head[None, :], Q_SIZE // 32),
        tl.where(live[:, None], sf_head, 0),
    )
    tl.store(
        scales
        + mxfp8_scale_offset(
            rows[:, None], Q_HEAD // 32 + g_tail[None, :], Q_SIZE // 32
        ),
        tl.where(live[:, None], sf_tail, 0),
        sf_tail_mask,
    )


@triton.jit
def _kv_norm_rows(kv, kvw, kvo, rows, live, kv_stride, eps, KV_SIZE: tl.constexpr):
    c_kv = tl.arange(0, KV_SIZE)
    x = tl.load(kv + rows[:, None] * kv_stride + c_kv[None, :], live[:, None], 0).to(
        tl.float32
    )
    w_kv = tl.load(kvw + c_kv).to(tl.float32)
    rrms_kv = tl.rsqrt(tl.sum(x * x, 1) / KV_SIZE + eps)
    y = x * rrms_kv[:, None] * w_kv[None, :]
    tl.store(kvo + rows[:, None] * KV_SIZE + c_kv[None, :], y, live[:, None])
    return y


@triton.jit(do_not_specialize=["num_tokens", "num_slots"])
def _q_kv_norm_quant_kernel(
    q,
    kv,
    qw,
    kvw,
    qo,
    kvo,
    scales,
    num_tokens,
    q_stride,
    kv_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    Q_HEAD: tl.constexpr,
    ROWS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    TRIGGER_FIRST: tl.constexpr = False,
    KV_ONLY: tl.constexpr = False,
    Q_TAIL_PAD: tl.constexpr = 0,
    cache=None,
    slots=None,
    positions=None,
    cos_sin=None,
    num_slots=0,
    block_stride=0,
    CACHE_BLOCK: tl.constexpr = 1,
    INSERT_KV: tl.constexpr = False,
    kv_scale=None,
    token_stride=0,
    KV_FP8: tl.constexpr = False,
):
    # ROWS tokens per program. Q is split at a power of two, Q_HEAD, so a
    # power-of-two tail does not pad either. With INSERT_KV
    # the grid's second axis splits the rows' Q (0) from their KV and its
    # cache insert (1), into a packed MXFP8 cache or, with KV_FP8, a
    # per-tensor FP8 one.
    Q_TAIL: tl.constexpr = Q_SIZE - Q_HEAD
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = rows < num_tokens
    if LAUNCH_PDL:
        # Dependents read the outputs only after their own grid-dependency
        # wait, so triggering first only lets them launch sooner: a decode
        # step's query projection then loads its weights while this kernel
        # waits for its input.
        if TRIGGER_FIRST:
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
        if not TRIGGER_FIRST:
            tl.extra.cuda.gdc_launch_dependents()
    if INSERT_KV:
        if tl.program_id(1) == 0:
            _q_norm_quant_rows(
                q,
                qw,
                qo,
                scales,
                rows,
                live,
                q_stride,
                eps,
                Q_SIZE,
                Q_HEAD,
                Q_TAIL,
                Q_TAIL_PAD,
                ROWS,
            )
        else:
            y = _kv_norm_rows(kv, kvw, kvo, rows, live, kv_stride, eps, KV_SIZE)
            if KV_FP8:
                _fp8_kv_insert(
                    y,
                    rows,
                    cache,
                    slots,
                    positions,
                    cos_sin,
                    kv_scale,
                    num_slots,
                    block_stride,
                    token_stride,
                    KV_SIZE,
                    CACHE_BLOCK,
                )
            else:
                _mxfp8_kv_insert(
                    y,
                    rows,
                    cache,
                    slots,
                    positions,
                    cos_sin,
                    num_slots,
                    block_stride,
                    KV_SIZE,
                    CACHE_BLOCK,
                )
    else:
        if not KV_ONLY:
            _q_norm_quant_rows(
                q,
                qw,
                qo,
                scales,
                rows,
                live,
                q_stride,
                eps,
                Q_SIZE,
                Q_HEAD,
                Q_TAIL,
                Q_TAIL_PAD,
                ROWS,
            )
        _kv_norm_rows(kv, kvw, kvo, rows, live, kv_stride, eps, KV_SIZE)


@triton.jit
def _norm_row(x, weight, row, stride, eps, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    w = tl.load(weight + cols, cols < SIZE, 0).to(tl.float32)
    values = tl.load(x + row * stride + cols, cols < SIZE, 0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(values * values, 0) / SIZE + eps)
    return values * rrms * w


@triton.jit(do_not_specialize=["num_tokens"])
def _q_kv_norm_quant_kernel_legacy(
    q,
    kv,
    qw,
    kvw,
    qo,
    kvo,
    scales,
    num_tokens,
    q_stride,
    kv_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    """The pre-port per-row kernel, kept verbatim for the default (bitwise) path."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(1) == 0:
        groups = tl.arange(0, BLOCK // 32)
        if row < num_tokens:
            y = _norm_row(q, qw, row, q_stride, eps, Q_SIZE, BLOCK)
            # Preserve the materialized normalization output's rounding boundary.
            y = y.to(q.dtype.element_ty).to(tl.float32)
            grouped = tl.reshape(y, (BLOCK // 32, 32))
            amax = tl.max(tl.abs(grouped), 1)
            normalized = amax * (1.0 / 448.0)
            bits = normalized.to(tl.uint32, bitcast=True)
            exponent = (bits >> 23) & 255
            mantissa = bits & 0x7FFFFF
            bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
            sf = tl.minimum(exponent + bump, 254)
            sf = tl.where(normalized <= 0, 0, sf)
            # Match FlashInfer's UE8M0 conversion, including zero/subnormal scales.
            inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
            inv_scale = inv_bits.to(tl.float32, bitcast=True)
            quantized = tl.reshape(grouped * inv_scale[:, None], (BLOCK,))
            tl.store(qo + row * Q_SIZE + cols, quantized, cols < Q_SIZE)
        else:
            sf = tl.full((BLOCK // 32,), 0, tl.uint32)
        padded_groups: tl.constexpr = triton.cdiv(Q_SIZE // 32, 4) * 4
        sf = tl.where(groups < Q_SIZE // 32, sf, 0)
        offsets = mxfp8_scale_offset(row, groups, Q_SIZE // 32)
        tl.store(scales + offsets, sf, groups < padded_groups)
    elif row < num_tokens:
        y = _norm_row(kv, kvw, row, kv_stride, eps, KV_SIZE, BLOCK)
        tl.store(kvo + row * KV_SIZE + cols, y, cols < KV_SIZE)


def _fused_q_kv_rmsnorm_quant_legacy(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Normalize Q/KV and quantize Q with FlashInfer's swizzled MXFP8 scales
    (the pre-port implementation, unchanged)."""
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    padded_tokens = triton.cdiv(tokens, 128) * 128
    scales = torch.empty(
        mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
    )
    if tokens:
        block = triton.next_power_of_2(max(q_size, kv_size))
        launch_pdl = current_platform.is_arch_support_pdl()
        _q_kv_norm_quant_kernel_legacy[(padded_tokens, 2)](
            qr,
            kv,
            q_weight,
            kv_weight,
            qo,
            kvo,
            scales,
            tokens,
            qr.stride(0),
            kv.stride(0),
            eps,
            q_size,
            kv_size,
            block,
            launch_pdl,
            num_warps=8 if block >= 2048 else 4,
            launch_pdl=launch_pdl,
        )
    return QuantizedActivation(qo, scales, qr.dtype, qr.shape, kMxfp8Dynamic), kvo


def fused_q_kv_rmsnorm_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    kv_insert: Mxfp8KVInsert | KVCacheInsert | None = None,
    early_trigger: bool = False,
    pad_rows: bool = True,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Normalize Q/KV, quantize Q to FlashInfer's MXFP8 and do ``kv_insert``
    (a packed MXFP8 or a per-tensor FP8 cache). With ``early_trigger``, the
    kernel triggers its PDL dependents before its own grid-dependency wait, for
    a caller whose query projection loads its weights before its wait. Without
    ``pad_rows``, the scales of the last 128-row atom's rows past the tokens
    stay unwritten, for consumers that compute rows independently.

    Unless VLLM_DSV41_FUSED_QKV_NORM_KV_INSERT is set (or a ``kv_insert`` /
    ``early_trigger`` / unpadded launch asks for the new kernel), this runs the
    per-row legacy kernel, bit for bit the pre-port output."""
    if (
        kv_insert is None
        and not early_trigger
        and pad_rows
        and not envs.VLLM_DSV41_FUSED_QKV_NORM_KV_INSERT
    ):
        return _fused_q_kv_rmsnorm_quant_legacy(qr, kv, q_weight, kv_weight, eps)
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    padded_tokens = triton.cdiv(tokens, 128) * 128
    # Rows past tokens (to the 128-row scale tile) stay unwritten; whole-tile
    # readers ignore them.
    qo = torch.empty(
        (padded_tokens, q_size), dtype=torch.float8_e4m3fn, device=qr.device
    )[:tokens]
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    scales = torch.empty(
        mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
    )
    if tokens:
        assert kv_size & (kv_size - 1) == 0
        q_head = 1 << ((q_size - 1).bit_length() - 1)
        # One warp per token row: 4 rows, 4 warps.
        rows = 4
        launch_pdl = current_platform.is_arch_support_pdl()
        insert: dict[str, object] = {}
        if kv_insert is not None:
            assert kv_insert.positions.shape[0] == tokens
            insert = dict(
                cache=kv_insert.cache,
                slots=kv_insert.slot_mapping,
                positions=kv_insert.positions,
                cos_sin=kv_insert.cos_sin_cache,
                num_slots=kv_insert.slot_mapping.shape[0],
                block_stride=kv_insert.cache.stride(0),
                CACHE_BLOCK=kv_insert.block_size,
                INSERT_KV=True,
            )
            if isinstance(kv_insert, KVCacheInsert):
                assert kv_insert.cache.dtype == torch.float8_e4m3fn
                insert.update(
                    kv_scale=kv_insert.scale,
                    token_stride=kv_insert.cache.stride(1),
                    KV_FP8=True,
                )
            else:
                assert kv_insert.cache.dtype == torch.uint8
        launched = padded_tokens if pad_rows else tokens
        _q_kv_norm_quant_kernel[(triton.cdiv(launched, rows), 2 if insert else 1)](
            qr,
            kv,
            q_weight,
            kv_weight,
            qo,
            kvo,
            scales,
            tokens,
            qr.stride(0),
            kv.stride(0),
            eps,
            q_size,
            kv_size,
            q_head,
            rows,
            launch_pdl,
            TRIGGER_FIRST=launch_pdl and early_trigger,
            Q_TAIL_PAD=triton.next_power_of_2(q_size - q_head),
            **insert,
            num_warps=4,
            launch_pdl=launch_pdl,
        )
    return QuantizedActivation(qo, scales, qr.dtype, qr.shape, kMxfp8Dynamic), kvo


def fused_kv_rmsnorm(
    kv: torch.Tensor, kv_weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """The KV half of :func:`fused_q_kv_rmsnorm_quant`, bit for bit."""
    assert kv.ndim == 2 and kv.stride(-1) == 1 and kv_weight.is_contiguous()
    tokens, kv_size = kv.shape
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    if tokens:
        assert kv_size & (kv_size - 1) == 0
        rows = 4
        launch_pdl = current_platform.is_arch_support_pdl()
        _q_kv_norm_quant_kernel[(triton.cdiv(tokens, rows),)](
            kv,
            kv,
            kv_weight,
            kv_weight,
            kvo,
            kvo,
            kvo,
            tokens,
            kv.stride(0),
            kv.stride(0),
            eps,
            kv_size,
            kv_size,
            kv_size,
            rows,
            launch_pdl,
            KV_ONLY=True,
            num_warps=4,
            launch_pdl=launch_pdl,
        )
    return kvo


def can_fuse_query_quant(linears: list[torch.nn.Module]) -> bool:
    """Require both local projections to use the same MXFP8 activation ABI."""
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    # QuantKey does not encode scale layout; restrict this producer to the
    # consumers that accept F8_128x4 swizzled scales.
    return all(
        get_input_quant_key(linear) == kMxfp8Dynamic
        and type(getattr(getattr(linear, "quant_method", None), "kernel", None))
        in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
        for linear in linears
    )
