# SPDX-License-Identifier: Apache-2.0
"""MOSCA IDX-FOLD: the DeepSeek-V4.1 decode index plumbing, folded (switch ``VLLM_MOSCA_IDX_FOLD``, default off).

Two folds, both pure integer plumbing, so bitwise to the served kernels by construction (and by test):

1. ``step_meta_fold``: ONE launch per step for the attention-metadata builds the runner issues before the decode graph.
   Served (per step, eager, on the critical path because the CPU runs ~7 ms ahead):
     per SWA KV-cache group  : torch.ge(slot_mapping, 0, out=is_valid_token); decode_swa_lens[T:] = 0;
                               _compute_swa_indices_and_lens_kernel                       (sparse_swa.py build)
     per compressed build    : [block_table.clamp_(min=0)]; out.fill_(-1); CompressedSlotMappingKernel
                               (FlashMLA builder, sparse_mla.py; indexer builder, indexer.py)
     per indexer build       : _indexer_decode_metadata_kernel; [seq_lens //= ratio]       (indexer.py, varlen decode)
   The builds are DATA: tables of tensors handed in by the caller (any number of groups / builds); nothing here knows
   how many groups a model has. The DeepGEMM paged-MQA schedule kernel that follows each indexer build is untouched.

2. ``remap_translate``: the sparse indexer's top-k column remap (``_sparse_topk_remap_kernel``, the producer of the
   request-local indices of a sparse index source) also writes the slot ids + lengths that
   ``compute_global_topk_indices_and_lens`` would compute from them (the translation every compressed layer runs in
   ``flash_mla_mega_attn._decode_compressed_kv_and_topk``), and the ``topk_indices_buffer[:n] = -1`` fill before it.

Every kernel here reproduces the served kernels' arithmetic in the served integer types; the docstrings name the
served line each step mirrors.
"""

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("VLLM_MOSCA_IDX_FOLD", "0") == "1"

SWA_COLS = 8      # slot, bt, bt_stride, replay, valid, idx, idx_stride, lens
CSM_COLS = 8      # bt, bt_stride, slot, out, block_size, ratio, clamp_rows, read_clamped
IDX_COLS = 12     # bt, bt_stride, out_seq_lens, out_bt, out_bt_stride, decode_lens, indices, per_req_lens, capacity,
#                   ratio, read_clamped, grid


@triton.jit
def _token_request(query_start_loc, token, num_reqs):
    # verbatim v1/attention/ops/metadata.py::_token_request
    lo = tl.full(token.shape, 0, tl.int32)
    hi = tl.full(token.shape, num_reqs, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + mid + 1, mid < num_reqs, other=0x7FFFFFFF)
        right = (lo < hi) & (end <= token)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    return tl.minimum(lo, num_reqs - 1)


@triton.jit(do_not_specialize=["G", "C", "I", "NT", "N", "T", "R", "R_DEC", "LMAX", "CSM_LEN", "N_TAIL", "N_FILL",
                               "GRID_I", "window_size", "swa_block", "BT_W", "index_width"])
def _mosca_step_meta_kernel(
    swa_tab, csm_tab, idx_tab,
    query_start_loc, seq_lens, token_to_req,
    G, C, I, NT, N, T, R, R_DEC, LMAX, CSM_LEN, N_TAIL, N_FILL, GRID_I,
    window_size, swa_block, BT_W, index_width,
    W_POW2: tl.constexpr, TB: tl.constexpr, BC_POW2: tl.constexpr, CSM_BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    s0 = G * NT
    s1 = s0 + G * N_TAIL
    s2 = s1 + C * R
    s3 = s2 + C * N_FILL
    if pid < s0:
        # ---- SWA build of group g for token t (sparse_swa.py DeepseekSparseSWAMetadataBuilder.build, decode, causal)
        g = pid // NT
        t = pid % NT
        row = swa_tab + g * 8
        slot_p = tl.load(row + 0).to(tl.pointer_type(tl.int64))
        bt_p = tl.load(row + 1).to(tl.pointer_type(tl.int32))
        bt_stride = tl.load(row + 2)
        replay_p = tl.load(row + 3).to(tl.pointer_type(tl.int32))
        valid_p = tl.load(row + 4).to(tl.pointer_type(tl.int8))
        idx_p = tl.load(row + 5).to(tl.pointer_type(tl.int32))
        idx_stride = tl.load(row + 6)
        lens_p = tl.load(row + 7).to(tl.pointer_type(tl.int32))
        # torch.ge(slot_mapping, 0, out=is_valid_token)
        is_valid = tl.load(slot_p + t, mask=t < N, other=-1) >= 0
        if t < N:
            tl.store(valid_p + t, is_valid.to(tl.int8))
        if t < T:
            offset = tl.arange(0, W_POW2)
            if is_valid == 0:
                # _compute_swa_indices_and_lens_kernel, invalid branch: len 0, row -1
                tl.store(lens_p + t, 0)
                tl.store(idx_p + t * idx_stride + offset, tl.full([W_POW2], -1, tl.int32), mask=offset < index_width)
            else:
                req_idx = tl.load(token_to_req + t)
                query_start = tl.load(query_start_loc + req_idx)
                query_end = tl.load(query_start_loc + req_idx + 1)
                query_len = query_end - query_start
                seq_len = tl.load(seq_lens + req_idx)
                prefix_len = seq_len - query_len
                pos = prefix_len + t - query_start
                start_pos = tl.maximum(pos - (window_size - 1), 0)  # left_add = max(0 - (w-1), 0) = 0 (no image)
                start_pos = tl.maximum(start_pos, tl.load(replay_p + req_idx))
                end_pos = pos + 1
                swa_len = end_pos - start_pos
                tl.store(lens_p + t, swa_len)
                pos_offset = start_pos + offset
                block_numbers = tl.load(bt_p + req_idx * bt_stride + pos_offset // swa_block,
                                        mask=pos_offset < end_pos, other=0)
                slot_ids = block_numbers * swa_block + pos_offset % swa_block
                slot_ids = tl.where(offset < swa_len, slot_ids, -1)
                tl.store(idx_p + t * idx_stride + offset, slot_ids, mask=offset < index_width)
    elif pid < s1:
        # ---- decode_swa_lens[num_decode_tokens:] = 0
        q = pid - s0
        g = q // N_TAIL
        j = q % N_TAIL
        lens_p = tl.load(swa_tab + g * 8 + 7).to(tl.pointer_type(tl.int32))
        offs = T + j * TB + tl.arange(0, TB)
        tl.store(lens_p + offs, tl.zeros([TB], tl.int32), mask=offs < LMAX)
    elif pid < s2:
        # ---- compressed slot mapping of request b for build c (+ the FlashMLA builder's block_table.clamp_(min=0))
        q = pid - s1
        c = q // R
        b = q % R
        row = csm_tab + c * 8
        bt_p = tl.load(row + 0).to(tl.pointer_type(tl.int32))
        bt_stride = tl.load(row + 1)
        slot_p = tl.load(row + 2).to(tl.pointer_type(tl.int64))
        out_p = tl.load(row + 3).to(tl.pointer_type(tl.int64))
        block_size = tl.load(row + 4).to(tl.int32)
        ratio = tl.load(row + 5).to(tl.int32)
        clamp_rows = tl.load(row + 6)
        read_clamped = tl.load(row + 7)
        if clamp_rows != 0:
            for i in range(0, BT_W, CSM_BLK):
                cols = i + tl.arange(0, CSM_BLK)
                v = tl.load(bt_p + b * bt_stride + cols, mask=cols < BT_W, other=0)
                tl.store(bt_p + b * bt_stride + cols, tl.maximum(v, 0), mask=cols < BT_W)
        query_start = tl.load(query_start_loc + b)
        query_end = tl.load(query_start_loc + b + 1)
        query_len = query_end - query_start
        seq_len = tl.load(seq_lens + b)
        start_pos = seq_len - query_len
        for i in range(0, query_len, CSM_BLK):
            offset = i + tl.arange(0, CSM_BLK)
            mask = offset < query_len
            pos = start_pos + i + tl.arange(0, CSM_BLK)
            slot = tl.load(slot_p + query_start + offset, mask=mask, other=-1)
            is_valid = ((pos + 1) % ratio == 0) & (slot != -1)
            pos_after_compress = pos // ratio  # dcp_world_size 1, cp_interleave 1 (asserted by the wrapper)
            block_ids = pos_after_compress // block_size
            block_numbers = tl.load(bt_p + b * bt_stride + block_ids, mask=mask & is_valid, other=0)
            if read_clamped != 0:
                block_numbers = tl.maximum(block_numbers, 0)
            slot_ids = block_numbers.to(tl.int64) * block_size + (pos_after_compress % block_size)
            slot_ids = tl.where(is_valid, slot_ids, -1)
            tl.store(out_p + query_start + offset, slot_ids, mask=mask)
    elif pid < s3:
        # ---- out.fill_(-1) of the compressed slot-mapping buffer beyond the tokens the requests cover
        q = pid - s2
        c = q // N_FILL
        j = q % N_FILL
        out_p = tl.load(csm_tab + c * 8 + 3).to(tl.pointer_type(tl.int64))
        covered = tl.load(query_start_loc + R)
        offs = j * TB + tl.arange(0, TB)
        tl.store(out_p + offs, tl.full([TB], -1, tl.int64), mask=(offs >= covered) & (offs < CSM_LEN))
    else:
        # ---- _indexer_decode_metadata_kernel of indexer build i, program `token` (+ seq_lens //= ratio)
        q = pid - s3
        i = q // GRID_I
        token = q % GRID_I
        row = idx_tab + i * 12
        grid_i = tl.load(row + 11)
        if token < grid_i:
            block_table = tl.load(row + 0).to(tl.pointer_type(tl.int32))
            input_stride = tl.load(row + 1)
            output_seq_lens = tl.load(row + 2).to(tl.pointer_type(tl.int32))
            output_block_table = tl.load(row + 3).to(tl.pointer_type(tl.int32))
            output_stride = tl.load(row + 4)
            decode_lens = tl.load(row + 5).to(tl.pointer_type(tl.int32))
            indices = tl.load(row + 6).to(tl.pointer_type(tl.int32))
            per_req_lens = tl.load(row + 7).to(tl.pointer_type(tl.int32))
            capacity = tl.load(row + 8)
            ratio = tl.load(row + 9).to(tl.int32)
            read_clamped = tl.load(row + 10)
            num_reqs = R_DEC
            num_tokens = T
            cols = tl.arange(0, BC_POW2)
            actual_tokens = tl.load(query_start_loc + num_reqs)
            if token < num_tokens:
                req = _token_request(query_start_loc, tl.full((1,), token, tl.int32), num_reqs)
                req = tl.sum(req, 0)
                valid = token < actual_tokens
                start = tl.load(query_start_loc + req)
                end = tl.load(query_start_loc + req + 1)
                seq = tl.load(seq_lens + req)
                length = tl.where(valid, seq - (end - start) + token - start + 1, 0)
                if ratio > 1:
                    length = length // ratio  # indexer.py: seq_lens //= self.compress_ratio (values >= 0)
                tl.store(output_seq_lens + token, length)
                tl.store(decode_lens + token, 1)
                tl.store(indices + token, tl.where(valid, req, num_reqs + token - actual_tokens))
                blocks = tl.load(block_table + req.to(tl.int64) * input_stride + cols, valid & (cols < BT_W), other=0)
                if read_clamped != 0:
                    blocks = tl.maximum(blocks, 0)
                tl.store(output_block_table + token.to(tl.int64) * output_stride + cols, blocks, cols < BT_W)
            else:
                tail = num_tokens + (token - num_tokens) * 256 + tl.arange(0, 256)
                tl.store(output_seq_lens + tail, tl.zeros([256], tl.int32), tail < capacity)
            if token < num_reqs:
                start = tl.load(query_start_loc + token)
                end = tl.load(query_start_loc + token + 1)
                tl.store(per_req_lens + token, end - start)


def _addr(x: torch.Tensor) -> int:
    return x.data_ptr()


_TAB_CACHE: dict = {}


def _table(rows: list[list[int]], ncols: int, device) -> torch.Tensor:
    key = (str(device), ncols, tuple(tuple(r) for r in rows))
    t = _TAB_CACHE.get(key)
    if t is None:
        t = torch.tensor(rows if rows else [[0] * ncols], dtype=torch.int64, device=device)
        _TAB_CACHE[key] = t
    return t


def step_meta_fold(swa_builds, csm_builds, idx_builds, query_start_loc, seq_lens, token_to_req, *, num_tokens,
                   num_decode_tokens, num_reqs, num_decodes, window_size, swa_block_size, index_width,
                   block_table_cols, lens_numel, csm_numel):
    """One launch for the step's decode metadata builds.

    swa_builds: list of dicts {slot_mapping [N] int64, block_table [R, *] int32, replay_start [R] int32,
                is_valid [>=N] bool, indices [>=T, index_width] int32 view, lens [lens_numel] int32}
    csm_builds: list of dicts {block_table, slot_mapping, out [csm_numel] int64, block_size, ratio, clamp_rows,
                read_clamped} in the served order (the build that clamps first)
    idx_builds: list of dicts {block_table, out_seq_lens [capacity], out_block_table [>=T, *], decode_lens, indices,
                per_req_lens, capacity, ratio, read_clamped}
    """
    dev = query_start_loc.device
    N, T, R = int(num_tokens), int(num_decode_tokens), int(num_reqs)
    swa_rows = [[_addr(b["slot_mapping"]), _addr(b["block_table"]), b["block_table"].stride(0),
                 _addr(b["replay_start"]), _addr(b["is_valid"]), _addr(b["indices"]), b["indices"].stride(0),
                 _addr(b["lens"])] for b in swa_builds]
    csm_rows = [[_addr(b["block_table"]), b["block_table"].stride(0), _addr(b["slot_mapping"]), _addr(b["out"]),
                 int(b["block_size"]), int(b["ratio"]), int(b["clamp_rows"]), int(b["read_clamped"])]
                for b in csm_builds]
    idx_rows = []
    for b in idx_builds:
        cap = int(b["capacity"])
        grid = max(int(num_decodes), T + triton.cdiv(cap - T, 256))
        idx_rows.append([_addr(b["block_table"]), b["block_table"].stride(0), _addr(b["out_seq_lens"]),
                         _addr(b["out_block_table"]), b["out_block_table"].stride(0), _addr(b["decode_lens"]),
                         _addr(b["indices"]), _addr(b["per_req_lens"]), cap, int(b["ratio"]),
                         int(b["read_clamped"]), grid])
    G, C, I = len(swa_rows), len(csm_rows), len(idx_rows)
    TB = 1024
    NT = max(N, T)
    n_tail = triton.cdiv(max(int(lens_numel) - T, 0), TB)
    n_fill = triton.cdiv(int(csm_numel), TB) if C else 0
    grid_i = max([r[11] for r in idx_rows], default=0)
    total = G * NT + G * n_tail + C * R + C * n_fill + I * grid_i
    if total == 0:
        return
    swa_tab = _table(swa_rows, SWA_COLS, dev)
    csm_tab = _table(csm_rows, CSM_COLS, dev)
    idx_tab = _table(idx_rows, IDX_COLS, dev)
    _mosca_step_meta_kernel[(total,)](
        swa_tab, csm_tab, idx_tab, query_start_loc, seq_lens, token_to_req,
        G, C, I, NT, N, T, R, int(num_decodes), int(lens_numel), int(csm_numel), n_tail, n_fill, grid_i,
        int(window_size), int(swa_block_size), int(block_table_cols), int(index_width),
        W_POW2=triton.next_power_of_2(int(index_width)), TB=TB,
        BC_POW2=triton.next_power_of_2(int(block_table_cols)), CSM_BLK=1024, num_warps=4,
    )


@triton.jit(do_not_specialize=["k", "width", "num_rows", "n_fill", "buf_w", "t_dec", "page"])
def _mosca_remap_translate_kernel(
    col_ptr, col_stride, si_ptr, si_stride, ks_ptr, ks_stride,
    local_ptr, local_stride, k, width, num_rows, n_fill, buf_w,
    glob_ptr, glob_stride, lens_ptr, tok2req_ptr, valid_ptr, bt_ptr, bt_stride, page, t_dec,
    SBK: tl.constexpr, W_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, W_POW2)
    # sparse_mqa_logits.py::_sparse_topk_remap_kernel (rows < num_rows, cols < k), after the
    # sparse_mqa_indexer.py `topk_indices_buffer[:n] = -1` fill (rows < n_fill, every buffer column)
    if row < num_rows:
        ks_base = tl.load(ks_ptr + row * ks_stride) // SBK
        c = tl.load(col_ptr + row * col_stride + cols, mask=cols < k, other=-1)
        valid = (c >= 0) & (c < width)
        cc = tl.where(valid, c, 0)
        blocks = tl.load(si_ptr + row * si_stride + cc // SBK, mask=cols < k, other=0)
        pos = (blocks - ks_base) * SBK + cc % SBK
        local = tl.where(valid & (cols < k), pos, -1)
    else:
        local = tl.full([W_POW2], -1, tl.int32)
    if row < n_fill:
        tl.store(local_ptr + row * local_stride + cols, local, mask=cols < buf_w)
    elif row < num_rows:
        tl.store(local_ptr + row * local_stride + cols, local, mask=cols < k)
    # cache_utils.py::_compute_global_topk_indices_and_lens_kernel over the buffer row (topk = buf_w)
    if row < t_dec:
        is_valid_token = tl.load(valid_ptr + row)
        req_idx = tl.load(tok2req_ptr + row)
        mask = cols < buf_w
        local_idx = tl.where(mask, local, -1)
        is_valid = local_idx >= 0
        block_numbers = tl.load(bt_ptr + req_idx * bt_stride + local_idx // page, mask=mask & is_valid, other=0)
        slot_ids = block_numbers * page + local_idx % page
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(glob_ptr + row * glob_stride + cols, slot_ids, mask=mask)
        count = tl.sum(is_valid.to(tl.int32), axis=0)
        tl.store(lens_ptr + row, tl.where(is_valid_token, count, 0))


def remap_translate(col_indices, sparse_indices, row_ks, sparse_block_kv, topk_tokens, width, topk_buffer, num_rows,
                    n_fill, token_to_req, is_valid_token, block_table, page, num_decode_tokens, out=None):
    """Folded producer of a sparse index source: writes topk_buffer[:max(num_rows, n_fill)] as the served fill + remap
    do, and returns (global slot ids [t_dec, buf_w], lens [t_dec]) equal to compute_global_topk_indices_and_lens of
    topk_buffer[:t_dec]."""
    buf_w = topk_buffer.shape[-1]
    t_dec = int(num_decode_tokens)
    assert buf_w <= 4096 and topk_tokens <= buf_w
    # the translation reads rows the fill or the remap wrote in this call
    assert t_dec <= max(int(num_rows), int(n_fill)), (t_dec, num_rows, n_fill)
    if out is None:
        glob = torch.empty((t_dec, buf_w), dtype=torch.int32, device=topk_buffer.device)
        lens = torch.empty((t_dec,), dtype=torch.int32, device=topk_buffer.device)
    else:
        glob, lens = out
    rows = max(int(num_rows), int(n_fill), t_dec)
    if rows == 0:
        return glob, lens
    cols = col_indices[: max(int(num_rows), 1)]
    _mosca_remap_translate_kernel[(rows,)](
        cols, cols.stride(0), sparse_indices, sparse_indices.stride(0), row_ks, row_ks.stride(0),
        topk_buffer, topk_buffer.stride(0), int(topk_tokens), int(width), int(num_rows), int(n_fill), buf_w,
        glob, glob.stride(0) if t_dec else buf_w, lens, token_to_req, is_valid_token, block_table,
        block_table.stride(0), int(page), t_dec,
        SBK=int(sparse_block_kv), W_POW2=triton.next_power_of_2(buf_w), num_warps=4,
    )
    return glob, lens
