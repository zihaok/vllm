# SPDX-License-Identifier: Apache-2.0
"""The eager lm_head tactic chosen per row count from a measured table (off unless VLLM_MOSCA_LMHEAD_TABLE is set).

The eager target lm_head is a BF16 GEMM x[M, K] @ W[N, K]^T (torch.nn.functional.linear, cuBLAS). On VR200 the cuBLAS
heuristic's tactic is the slowest of the candidates over parts of the row band at concurrency 6144 (nvjet 128x480 at
M 1764-1920, and 128x352 above 1932, while 256x128 or 192x192 tiles finish 3-11 % sooner). This module runs the same GEMM
through cuBLASLt with the candidate the table names for M (all candidates are non-split-K and bitwise equal to the
served call, checked at every row) and otherwise leaves the served F.linear in place.

VLLM_MOSCA_LMHEAD_TABLE = path of a JSON file:
  {"N": 129280, "K": 5120, "pick_step": 12, "cc": [10, 7], "sms": 212,
   "lo": 1674, "hi": 2046,
   "rows": {"<M_p>": {"tile": <TILE_ID>, "cluster": <CLUSTER_SHAPE_ID>, "stages": <STAGES_ID>, "custom": <CUSTOM_OPTION>,
                      "tile_rows": <row-tile height of that tactic>}}}
A row M in [lo, hi] uses the entry of pick row pick_step * floor(M / pick_step) while M needs the same number of row tiles
(tile_rows) of that entry's tactic as the pick row, else the next pick row up under the same test (a tactic's cost
steps at every multiple of its tile height). No entry, a device or shape mismatch, no
heuristic candidate with that config, or any cuBLASLt error -> None, and the caller runs the served F.linear.
"""

import ctypes
import json
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CFG_ATTRS = (("tile", 1), ("stages", 6), ("custom", 5), ("cluster", 8))


class _HeurResult(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_uint64 * 8), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


def _loaded_lib(sub: str) -> str | None:
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and sub in parts[-1]:
                return parts[-1]
    return None


class MoscaLmHeadLt:
    WS_BYTES = 32 << 20

    def __init__(self, table_path: str):
        with open(table_path) as f:
            t = json.load(f)
        self.N, self.K, self.step = int(t["N"]), int(t["K"]), int(t.get("pick_step", 12))
        self.lo, self.hi = int(t.get("lo", 0)), int(t.get("hi", 1 << 30))
        self.rows = {int(k): v for k, v in t["rows"].items()}
        self.cc = tuple(t["cc"]) if t.get("cc") else None
        self.sms = t.get("sms")
        self.ready = None  # lazily: cuBLASLt handle and descriptors
        self.algo = {}     # M -> HeurResult or False
        self.lay = {}

    def _init(self, x: torch.Tensor) -> bool:
        if self.ready is not None:
            return self.ready
        self.ready = False
        props = torch.cuda.get_device_properties(x.device)
        if (self.cc and (props.major, props.minor) != self.cc) or (self.sms and props.multi_processor_count != self.sms):
            logger.warning_once("MOSCA lm_head table is for cc %s / %s SMs; keeping F.linear.", self.cc, self.sms)
            return False
        path = _loaded_lib("libcublasLt.so")
        if path is None:
            logger.warning_once("MOSCA lm_head: libcublasLt not loaded; keeping F.linear.")
            return False
        L = self.L = ctypes.CDLL(path)
        self.h = ctypes.c_void_p()
        self.desc = ctypes.c_void_p()
        op_t, op_n = ctypes.c_int32(1), ctypes.c_int32(0)
        ok = (L.cublasLtCreate(ctypes.byref(self.h)) == 0
              and L.cublasLtMatmulDescCreate(ctypes.byref(self.desc), ctypes.c_int(68), ctypes.c_int(0)) == 0
              and L.cublasLtMatmulDescSetAttribute(self.desc, ctypes.c_int(3), ctypes.byref(op_t), ctypes.c_size_t(4)) == 0
              and L.cublasLtMatmulDescSetAttribute(self.desc, ctypes.c_int(4), ctypes.byref(op_n), ctypes.c_size_t(4)) == 0)
        if not ok:
            logger.warning_once("MOSCA lm_head: cuBLASLt init failed; keeping F.linear.")
            return False
        self.ws = torch.empty(self.WS_BYTES, dtype=torch.uint8, device=x.device)
        self.ready = True
        return True

    def _layouts(self, M: int):
        if M not in self.lay:
            L, a, b, c = self.L, ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            # column-major view of torch's call: D[N, M] = op_T(W)[N, K] x X[K, M], all BF16, fp32 accumulate
            L.cublasLtMatrixLayoutCreate(ctypes.byref(a), ctypes.c_int(14), ctypes.c_uint64(self.K), ctypes.c_uint64(self.N), ctypes.c_int64(self.K))
            L.cublasLtMatrixLayoutCreate(ctypes.byref(b), ctypes.c_int(14), ctypes.c_uint64(self.K), ctypes.c_uint64(M), ctypes.c_int64(self.K))
            L.cublasLtMatrixLayoutCreate(ctypes.byref(c), ctypes.c_int(14), ctypes.c_uint64(self.N), ctypes.c_uint64(M), ctypes.c_int64(self.N))
            self.lay[M] = (a, b, c)
        return self.lay[M]

    def _cfg(self, r) -> dict:
        out = {}
        for name, attr in _CFG_ATTRS:
            wr = ctypes.c_size_t(0)
            self.L.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(r.algo), ctypes.c_int(attr), None, ctypes.c_size_t(0), ctypes.byref(wr))
            buf = (ctypes.c_uint8 * max(1, wr.value))()
            st = self.L.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(r.algo), ctypes.c_int(attr), buf, ctypes.c_size_t(wr.value), ctypes.byref(wr))
            out[name] = int.from_bytes(bytes(buf), "little") if st == 0 else None
        return out

    def _entry(self, M: int):
        """The table entry for M: the pick row at or below M, used only while M needs the same number of row tiles of
        that entry's tactic as its pick row (a tactic's cost steps at every multiple of its tile height); else the next
        pick row up under the same test. A pick row without an entry means the served call."""
        if not (self.lo <= M <= self.hi):
            return None
        mp = self.step * (M // self.step)
        for cand in (mp, mp + self.step):
            e = self.rows.get(cand)
            if e is None:
                return None
            tr = int(e.get("tile_rows") or 0)
            if tr > 0 and -(-M // tr) == -(-cand // tr):
                return e
        return None

    def _pick(self, M: int):
        if M in self.algo:
            return self.algo[M]
        if torch.cuda.is_current_stream_capturing():
            return False  # never query heuristics inside a capture; not cached
        want = self._entry(M)
        self.algo[M] = False
        if want is None:
            return False
        a, b, c = self._layouts(M)
        pref = ctypes.c_void_p()
        self.L.cublasLtMatmulPreferenceCreate(ctypes.byref(pref))
        wsz = ctypes.c_uint64(self.WS_BYTES)
        self.L.cublasLtMatmulPreferenceSetAttribute(pref, ctypes.c_int(1), ctypes.byref(wsz), ctypes.c_size_t(8))
        res = (_HeurResult * 16)()
        cnt = ctypes.c_int(0)
        st = self.L.cublasLtMatmulAlgoGetHeuristic(self.h, self.desc, a, b, c, c, pref, ctypes.c_int(16), res, ctypes.byref(cnt))
        self.L.cublasLtMatmulPreferenceDestroy(pref)
        if st == 0:
            for i in range(cnt.value):
                cfg = self._cfg(res[i])
                if all(cfg.get(k) == want.get(k) for k in ("tile", "cluster", "stages", "custom")) and res[i].workspaceSize <= self.WS_BYTES:
                    self.algo[M] = res[i]
                    break
        if self.algo[M] is False:
            logger.warning_once("MOSCA lm_head: no heuristic candidate matches the table at M=%d; F.linear there.", M)
        return self.algo[M]

    def maybe(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
        """The lm_head GEMM with the table's tactic, or None (the caller then runs the served F.linear)."""
        if (x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or not x.is_contiguous()
                or not weight.is_contiguous() or tuple(weight.shape) != (self.N, self.K) or x.shape[1] != self.K):
            return None
        if not self._init(x):
            return None
        M = x.shape[0]
        r = self._pick(M)
        if not r:
            return None
        out = torch.empty((M, self.N), dtype=torch.bfloat16, device=x.device)
        a, b, c = self._layouts(M)
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        st = self.L.cublasLtMatmul(self.h, self.desc, ctypes.byref(alpha), ctypes.c_void_p(weight.data_ptr()), a,
                                   ctypes.c_void_p(x.data_ptr()), b, ctypes.byref(beta), ctypes.c_void_p(out.data_ptr()), c,
                                   ctypes.c_void_p(out.data_ptr()), c, ctypes.byref(r.algo), ctypes.c_void_p(self.ws.data_ptr()),
                                   ctypes.c_size_t(self.WS_BYTES), ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream))
        if st != 0:
            logger.warning_once("MOSCA lm_head: cublasLtMatmul status %d; F.linear from now on at M=%d.", st, M)
            self.algo[M] = False
            return None
        return out


_INSTANCE: MoscaLmHeadLt | None | bool = None


def get() -> MoscaLmHeadLt | None:
    global _INSTANCE
    if _INSTANCE is None:
        path = os.environ.get("VLLM_MOSCA_LMHEAD_TABLE", "")
        try:
            _INSTANCE = MoscaLmHeadLt(path) if path else False
        except Exception as e:  # a bad table keeps the served path
            logger.warning_once("MOSCA lm_head table %s not used: %s", path, e)
            _INSTANCE = False
    return _INSTANCE or None
