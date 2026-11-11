"""Python entry of the MOSCA sparse-MLA decode kernel (mosca_smla.cu), with the argument list of the specialized-decode
wrapper ``trtllm_decode_sparse_mla_dsv4`` (sparse_indices are KV storage offsets).

``admit(...)`` returns None when the kernel takes the call, otherwise a short reason (the census word); ``run(...)`` launches it.
The split table (columns W -> cluster size S, keys per CTA KPC) is DATA: ``split_table.json`` next to this file, overridable with
``MOSCA_SMLA_SPLIT_TABLE``.
"""
import ctypes
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None
SWA_WIDTH = 128
NH, HD = 16, 512
MERGE = {"dsmem": 0, "global": 1, "dsmem16": 2}
QK = {"tile": 0, "ksplit": 1}
MAX_ROWS_SCRATCH = 64
_SCRATCH = {}


def scratch(device):
    """The global-merge scratch of a device ([rows <= 64][S <= 16][16][512] FP32 + (m, l)), allocated once outside any capture."""
    buf = _SCRATCH.get(device)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mosca_smla: the global-merge scratch must be allocated before graph capture (run once eagerly)")
        ws = torch.empty(MAX_ROWS_SCRATCH * 16 * NH * HD, dtype=torch.float32, device=device)
        ml = torch.empty(MAX_ROWS_SCRATCH * 16 * 2 * NH, dtype=torch.float32, device=device)
        buf = _SCRATCH[device] = (ws, ml)
    return buf


def lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("MOSCA_SMLA_LIB", os.path.join(HERE, "libmosca_smla.so"))
        L = ctypes.CDLL(path)
        L.mosca_smla_launch.restype = ctypes.c_int
        L.mosca_smla_launch.argtypes = (
            [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong]
            + [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong] * 2
            + [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int]
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            + [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
            + [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong]
            + [ctypes.c_int] * 6 + [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
        L.mosca_smla_max_active_clusters.restype = ctypes.c_int
        L.mosca_smla_max_active_clusters.argtypes = [ctypes.c_int, ctypes.c_int]
        L.mosca_smla_smem_bytes.restype = ctypes.c_int
        L.mosca_smla_smem_bytes.argtypes = [ctypes.c_int]
        L.mosca_smla_uses_alias.restype = ctypes.c_int
        L.mosca_smla_uses_alias.argtypes = [ctypes.c_int]
        _LIB = L
    return _LIB


_TABLE = None


def split_table():
    global _TABLE
    if _TABLE is None:
        path = os.environ.get("MOSCA_SMLA_SPLIT_TABLE", os.path.join(HERE, "split_table.json"))
        with open(path) as fh:
            _TABLE = [(int(r["width"]), int(r["max_rows"]), int(r["S"]), int(r["KPC"]), MERGE[r.get("merge", "dsmem")],
                       QK[r.get("qk", "tile")], int(r.get("chunk", 0))) for r in json.load(fh)["rules"]]
    return _TABLE


def pick_split(width, rows, table=None, generic=True):
    """(S, KPC, merge, qk, chunk) for an index width and row count: the table's first matching rule, else the smallest power-of-two S <= 16
    whose KPC fits 128 keys with the DSMEM merge and the tile QK (None above 2048 columns)."""
    table = split_table() if table is None else table
    for w, max_rows, s, kpc, merge, qk, chunk in table:
        if w == width and rows <= max_rows:
            return s, kpc, merge, qk, chunk
    if not generic:
        return None
    for s in (2, 4, 8, 16):
        kpc = -(-width // s)
        kpc = -(-kpc // 8) * 8
        if kpc <= 128:
            return s, kpc, 0, 0, 0
    return None


def _pool_tokens(cache):
    return cache.shape[0] * cache.stride(0) // cache.stride(-2)


_CENSUS = {}
_LOG = None


def census(rows, width, reason):
    """Count the calls this process routes (reason None = the MOSCA kernel; else the reason it keeps FlashInfer); one log line per
    key at its first call and at every 40th (the executor's graph captures run each layer once per captured size)."""
    global _LOG
    key = (int(rows), int(width), reason or "admitted")
    n = _CENSUS.get(key, 0) + 1
    _CENSUS[key] = n
    if n == 1 or n % 40 == 0:
        if _LOG is None:
            try:
                from vllm.logger import init_logger
                _LOG = init_logger("vllm.mosca.sparse_mla_decode")
            except Exception:  # noqa: BLE001
                import logging
                _LOG = logging.getLogger("mosca.sparse_mla_decode")
        _LOG.info("MOSCA sparse-MLA decode census: rows=%d width=%d route=%s calls=%d", key[0], key[1], key[2], n)


def lib_ok():
    """Whether the kernel library loads (typed reason otherwise)."""
    try:
        lib()
        return None
    except OSError as e:
        return "lib_load_failed:%s" % type(e).__name__


def admit(query, swa_kv_cache, compressed_kv_cache, sparse_indices, sparse_topk_lens, seq_lens, out, bmm1_scale, bmm2_scale, sinks,
          cum_seq_lens_q):
    """None when the MOSCA kernel takes this call, else the reason it keeps the FlashInfer kernel."""
    bad = lib_ok()
    if bad:
        return bad
    if cum_seq_lens_q is None:
        return "no cum_seq_lens_q (DSpark non-causal path)"
    if query.dim() != 3 or tuple(query.shape[1:]) != (NH, HD) or query.dtype != torch.float8_e4m3fn or query.stride(-1) != 1:
        return "query is not FP8 [T, 16, 512]"
    if out.dim() != 3 or tuple(out.shape[1:]) != (NH, HD) or out.dtype != torch.bfloat16 or out.stride(-1) != 1:
        return "out is not BF16 [T, 16, 512]"
    for c in (swa_kv_cache, compressed_kv_cache):
        if c.dtype != torch.float8_e4m3fn or c.shape[-1] != HD or c.stride(-1) != 1 or c.stride(-2) < HD:
            return "KV cache is not FP8 rows of 512"
    if sparse_indices.dtype != torch.int32 or sparse_indices.dim() != 2 or sparse_indices.stride(-1) != 1:
        return "indices are not int32 [T, W]"
    if sparse_indices.shape[1] < SWA_WIDTH:
        return "index width below the SWA width"
    if pick_split(sparse_indices.shape[1], query.shape[0], generic=os.environ.get("MOSCA_SMLA_TABLE_ONLY", "1") != "1") is None:
        return "no split rule for (width, rows)"
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        return "tensor scales"
    if sinks is not None and (sinks.dtype != torch.float32 or sinks.numel() < NH):
        return "sinks are not FP32 [16]"
    for t in (sparse_topk_lens, seq_lens, cum_seq_lens_q):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata is not contiguous int32"
    return None


def run(query, swa_kv_cache, compressed_kv_cache, sparse_indices, sparse_topk_lens, seq_lens, out, bmm1_scale, bmm2_scale, sinks,
        cum_seq_lens_q, split=None, pdl=True):
    T = query.shape[0]
    W = sparse_indices.shape[1]
    pick = pick_split(W, T) if split is None else tuple(split)
    S, KPC = pick[0], pick[1]
    merge = pick[2] if len(pick) > 2 else 0
    qk = pick[3] if len(pick) > 3 else 0
    chunk = pick[4] if len(pick) > 4 else 0
    ws = ml = None
    if merge == 1:
        if T > MAX_ROWS_SCRATCH:
            raise RuntimeError("mosca_smla: global merge above %d rows" % MAX_ROWS_SCRATCH)
        ws, ml = scratch(query.device)
    B = cum_seq_lens_q.numel() - 1
    stream = torch.cuda.current_stream(query.device).cuda_stream
    rc = lib().mosca_smla_launch(
        query.data_ptr(), query.stride(0), query.stride(1),
        swa_kv_cache.data_ptr(), swa_kv_cache.stride(-2), _pool_tokens(swa_kv_cache),
        compressed_kv_cache.data_ptr(), compressed_kv_cache.stride(-2), _pool_tokens(compressed_kv_cache),
        sparse_indices.data_ptr(), sparse_indices.stride(0), W,
        sparse_topk_lens.data_ptr(), seq_lens.data_ptr(), cum_seq_lens_q.data_ptr(), B,
        sinks.data_ptr() if sinks is not None else None, float(bmm1_scale), float(bmm2_scale),
        out.data_ptr(), out.stride(0), out.stride(1),
        T, S, KPC, SWA_WIDTH, 1 if pdl else 0, merge, None if ws is None else ws.data_ptr(), None if ml is None else ml.data_ptr(),
        qk, chunk, stream)
    if rc != 0:
        raise RuntimeError("mosca_smla_launch failed: %d (S=%d KPC=%d merge=%d qk=%d chunk=%d W=%d T=%d)" % (rc, S, KPC, merge, qk, chunk, W, T))
    return out
