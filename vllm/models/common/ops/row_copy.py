# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vectorized row copies that PyTorch runs through its slow generic kernels.

``x.unsqueeze(1).expand(-1, n, -1).contiguous()`` (and ``.repeat``) reads a
stride-0 source, and ``Tensor.index_copy_`` copies element by element, so
both run PyTorch's non-vectorized element-wise kernels at ~1 TB/s. These
Triton kernels move whole rows with wide loads and stores instead. They copy
bits only, so the results are identical.
"""

import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BLOCK = 4096


@triton.jit
def _broadcast_rows_kernel(
    src_ptr,
    src_stride,
    dst_ptr,
    dst_stride_0,
    dst_stride_1,
    row_len,
    NUM_COPIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_len
    x = tl.load(src_ptr + row * src_stride + offs, mask=mask)
    for i in tl.static_range(NUM_COPIES):
        tl.store(dst_ptr + row * dst_stride_0 + i * dst_stride_1 + offs, x, mask=mask)


def broadcast_rows(x: torch.Tensor, num_copies: int) -> torch.Tensor:
    """``x.unsqueeze(1).expand(-1, num_copies, -1).contiguous()`` for 2D x."""
    assert x.dim() == 2
    if not current_platform.is_cuda_alike() or x.stride(-1) != 1 or x.numel() == 0:
        return x.unsqueeze(1).expand(-1, num_copies, -1).contiguous()
    num_rows, row_len = x.shape
    out = x.new_empty(num_rows, num_copies, row_len)
    _broadcast_rows_kernel[(num_rows, triton.cdiv(row_len, _BLOCK))](
        x,
        x.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        row_len,
        NUM_COPIES=num_copies,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def _index_copy_rows_kernel(
    dst_ptr,
    dst_stride,
    src_ptr,
    src_stride,
    rows_ptr,
    row_len,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_len
    dst_row = tl.load(rows_ptr + i).to(tl.int64)
    x = tl.load(src_ptr + i * src_stride + offs, mask=mask)
    tl.store(dst_ptr + dst_row * dst_stride + offs, x, mask=mask)


def index_copy_rows_(
    out: torch.Tensor, rows: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    """``out.index_copy_(0, rows, src)``: out[rows[i]] = src[i].

    Like index_copy_, ``rows`` must not repeat (the winner among duplicates
    is unspecified in both).
    """
    assert rows.dim() == 1 and src.shape[0] == rows.shape[0]
    assert out.shape[1:] == src.shape[1:] and out.dtype == src.dtype
    row_len = math.prod(out.shape[1:])
    if (
        not current_platform.is_cuda_alike()
        or rows.numel() == 0
        or row_len == 0
        or not _rows_contiguous(out)
        or not _rows_contiguous(src)
    ):
        return out.index_copy_(0, rows, src)
    _index_copy_rows_kernel[(rows.shape[0], triton.cdiv(row_len, _BLOCK))](
        out,
        out.stride(0),
        src,
        src.stride(0),
        rows,
        row_len,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out


def _rows_contiguous(t: torch.Tensor) -> bool:
    """Each row (dim 0 slice) is one contiguous run of memory."""
    return t.dim() >= 1 and t[0:1].is_contiguous()
