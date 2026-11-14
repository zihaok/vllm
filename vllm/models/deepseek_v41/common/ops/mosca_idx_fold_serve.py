# SPDX-License-Identifier: Apache-2.0
"""MOSCA IDX-FOLD serve wiring (switch ``VLLM_MOSCA_IDX_FOLD``, default off).

- ``StepMetaCollector``: the runner's ``build_attn_metadata`` asks every builder that implements ``mosca_fold_builds`` for the
  decode-metadata kernels its ``build`` would launch this step, in the served order, and launches them as ONE kernel
  (``mosca_idx_fold.step_meta_fold``); the builders' ``build`` then skip those kernels and return the same metadata views.
- ``armed``: the mega attention layer of a sparse index source arms its indexer call with the translation context, so the
  sparse indexer's remap also writes the slot ids + lengths (``mosca_idx_fold.remap_translate``) and publishes them.

Census (always on with the switch; typed log lines):
  ``IDXFOLD_CENSUS rank=<r> launches=<n> declined=<n> builds=swa<k>,csm<k>,idx<k> capture=<T:tr/x1/prod/rt;...>``
  at the first fold launch, at every verify step and every 4096 launches. ``capture`` is what each captured decode graph
  contains: tr = served translations, x1 = consumer reuses, prod = index sources fed by their producer, rt = folded remaps.

``VLLM_MOSCA_IDX_FOLD_VERIFY=N`` (served-entry proof, fails closed):
  - metadata: at the first N fold launches (then every 512th, 32 more), the fold's outputs are snapshot, the served kernels
    run as well (the builders do not skip) and every buffer is compared;
  - translations: every reused (slot ids, lens) is compared on the GPU with the served translation of the same inputs, and every
    folded remap's topk_indices_buffer rows with the served fill + remap of the same DeepSelect output, inside the captured
    graphs too (device counters: checks, mismatching elements), read at each verify step.
  Each verify step logs ``IDXFOLD_VERIFY rank=<r> step=<k> meta_buffers=<n> meta_mismatch=<n> xlate_checks=<n> xlate_mismatch=<n>``;
  a mismatch, or no translation check by the last window step (``verify_vacuous``), logs ``IDXFOLD_VERIFY_FAIL <word>`` and raises.
"""

import contextlib
import os

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.ops import mosca_idx_fold as F

logger = init_logger(__name__)

ENABLED = F.ENABLED
VERIFY_STEPS = int(os.environ.get("VLLM_MOSCA_IDX_FOLD_VERIFY", "0"))
CENSUS_EVERY = 4096
VERIFY_LOG = {"launches": 0, "declined": 0, "served": 0, "meta_steps": 0, "meta_mismatch": 0}
SERVED_WHY: dict = {}   # step-level declines by word (no_decode_shape, hookless_builder)
CAPTURE_CENSUS: dict[int, dict[str, set]] = {}
ARMED = None
_DEV_COUNTERS: dict[str, torch.Tensor] = {}


def _rank() -> str:
    """dp<DP rank>.tp<TP rank> (one decode rank per DP engine under DEP), else the pid."""
    try:
        from vllm.distributed.parallel_state import get_dp_group, get_tp_group

        return f"dp{get_dp_group().rank_in_group}.tp{get_tp_group().rank_in_group}"
    except Exception:  # noqa: BLE001
        return f"pid{os.getpid()}"


def census_add(num_decode_tokens: int, key: str, ident) -> None:
    """Python-level census of what a captured decode graph contains (recorded only while capturing). Per captured size, the set
    of layers (or index sources) that took each path, so a size captured twice counts each layer once."""
    if torch.cuda.is_current_stream_capturing():
        c = CAPTURE_CENSUS.setdefault(int(num_decode_tokens), {"tr": set(), "x1": set(), "prod": set(), "rt": set()})
        c[key].add(ident)


@contextlib.contextmanager
def armed(ctx):
    global ARMED
    prev, ARMED = ARMED, ctx
    try:
        yield
    finally:
        ARMED = prev


def _counters(device) -> torch.Tensor:
    key = str(device)
    t = _DEV_COUNTERS.get(key)
    if t is None:
        t = torch.zeros(2, dtype=torch.int64, device=device)
        _DEV_COUNTERS[key] = t
    return t


def verify_xlate(indices, lens, served_fn) -> None:
    """VERIFY: compare a reused translation with the served one on the GPU (graph-safe: no host sync)."""
    ref_i, ref_l = served_fn()
    verify_tensors((ref_i, indices), (ref_l, lens))


def verify_tensors(*pairs) -> None:
    """VERIFY: count one check and the mismatching elements of (served, folded) pairs in device counters."""
    c = _counters(pairs[0][1].device)
    c[0:1].add_(1)
    d = None
    for ref, got in pairs:
        x = (ref != got).sum().view(1)
        d = x if d is None else d + x
    c[1:2].add_(d)


def _fail(word: str, detail: str) -> None:
    line = f"IDXFOLD_VERIFY_FAIL {word} rank={_rank()} {detail}"
    print(line, flush=True)
    raise RuntimeError(line)


def _census_line(collector=None) -> str:
    cap = ";".join(f"{t}:{len(v['tr'])}/{len(v['x1'])}/{len(v['prod'])}/{len(v['rt'])}" for t, v in sorted(CAPTURE_CENSUS.items()))
    b = f"swa{len(collector.swa)},csm{len(collector.csm)},idx{len(collector.idx)}" if collector is not None else "-"
    why = ",".join(f"{k}:{v}" for k, v in sorted(SERVED_WHY.items())) or "-"
    return (f"IDXFOLD_CENSUS rank={_rank()} launches={VERIFY_LOG['launches']} declined={VERIFY_LOG['declined']} "
            f"served={VERIFY_LOG['served']}:{why} builds={b} capture={cap or '-'}")


def _verify_step(i: int) -> bool:
    if VERIFY_STEPS <= 0:
        return False
    return i < VERIFY_STEPS or (i % 512 == 0 and i // 512 <= 32)


class StepMetaCollector:
    """Collects the step's foldable builds in the served order and launches them as one kernel."""

    def __init__(self):
        self.swa, self.csm, self.idx = [], [], []
        self.builders = []
        self.common = None
        self.clamped = set()
        self.shape = {}
        self.snap = None
        self.step = None

    @property
    def decode_tokens(self):
        return self.shape["T"]

    def has_decode_shape(self) -> bool:
        """True when a decode-aware build (SWA, indexer) defined the step's decode tokens."""
        return "T" in self.shape

    def release(self, why: str) -> None:
        """The step runs served whole: every builder that offered a build builds its own kernels; typed in the census."""
        for b in self.builders:
            b._mosca_folded = False
        VERIFY_LOG["served"] += 1
        SERVED_WHY[why] = SERVED_WHY.get(why, 0) + 1
        if VERIFY_LOG["served"] == 1:
            print(_census_line(self) + f" served_why={why}", flush=True)

    @property
    def decodes(self):
        return self.shape.get("num_decodes", 0)

    @property
    def window_size(self):
        return self.shape.get("window", 0)

    @property
    def swa_block_size(self):
        return self.shape.get("swa_block", 1)

    @property
    def index_width(self):
        return self.shape.get("index_width", 1)

    def step_idx(self):
        return VERIFY_LOG["launches"]

    def _shape(self, b, *keys):
        for k in keys:
            v = b.pop("_" + k, None)
            if v is None:
                continue
            if self.shape.setdefault(k, v) != v:
                raise RuntimeError(f"MOSCA IDX-FOLD: builds disagree on {k}: {self.shape[k]} vs {v}")

    def add(self, builder, kind_builds, cm, token_to_req):
        if kind_builds is None:
            return
        key = (cm.query_start_loc.data_ptr(), cm.seq_lens.data_ptr(), token_to_req.data_ptr(), cm.num_actual_tokens,
               cm.num_reqs)
        if self.common is None:
            self.common = dict(key=key, query_start_loc=cm.query_start_loc, seq_lens=cm.seq_lens, token_to_req=token_to_req)
        elif self.common["key"] != key:
            # a group with other batch tensors: leave its kernels to its own build (served path)
            builder._mosca_folded = False
            VERIFY_LOG["declined"] += 1
            return
        for kind, b in kind_builds:
            self._shape(b, "T", "N", "num_decodes", "window", "swa_block", "index_width")
            if kind == "csm":
                bt = b["block_table"].data_ptr()
                if b.get("clamp_rows"):
                    self.clamped.add(bt)
                b["read_clamped"] = 1 if bt in self.clamped else 0
                self.csm.append(b)
            elif kind == "idx":
                b["read_clamped"] = 1 if b["block_table"].data_ptr() in self.clamped else 0
                self.idx.append(b)
            else:
                self.swa.append(b)
        self.builders.append(builder)

    def launch(self, num_tokens, num_decode_tokens, num_reqs, num_decodes, window_size, swa_block_size, index_width,
               block_table_cols, lens_numel, csm_numel, step_idx):
        if not self.builders:
            return
        c = self.common
        F.step_meta_fold(self.swa, self.csm, self.idx, c["query_start_loc"], c["seq_lens"], c["token_to_req"],
                         num_tokens=num_tokens, num_decode_tokens=num_decode_tokens, num_reqs=num_reqs, num_decodes=num_decodes,
                         window_size=window_size, swa_block_size=swa_block_size, index_width=index_width,
                         block_table_cols=block_table_cols, lens_numel=lens_numel, csm_numel=csm_numel)
        i = VERIFY_LOG["launches"]
        VERIFY_LOG["launches"] += 1
        self.step = i
        if i == 0 or i % CENSUS_EVERY == 0:
            print(_census_line(self), flush=True)
        if _verify_step(i):
            self.snap = [(t, t.clone()) for t in self.outputs()]
            for b in self.builders:
                b._mosca_folded = False  # the served kernels run too; finish() compares

    def outputs(self):
        out = []
        for b in self.swa:
            out += [b["is_valid"], b["indices"], b["lens"]]
        for b in self.csm:
            out += [b["out"], b["block_table"]]
        for b in self.idx:
            out += [b["out_seq_lens"], b["out_block_table"], b["decode_lens"], b["indices"], b["per_req_lens"]]
        return out

    def finish(self):
        if self.snap is None:
            return
        VERIFY_LOG["meta_steps"] += 1
        bad = sum(0 if torch.equal(t, s) else 1 for t, s in self.snap)
        VERIFY_LOG["meta_mismatch"] += bad
        dev = self.snap[0][0].device
        checks, mism = (int(x) for x in _counters(dev).tolist())
        line = (f"IDXFOLD_VERIFY rank={_rank()} step={self.step} meta_buffers={len(self.snap)} meta_mismatch={bad} "
                f"xlate_checks={checks} xlate_mismatch={mism}")
        print(line, flush=True)
        print(_census_line(self), flush=True)
        if bad:
            _fail("served_entry_mismatch:meta", line)
        if mism:
            _fail("served_entry_mismatch:xlate", line)
        if self.step == VERIFY_STEPS - 1 and checks == 0:
            _fail("verify_vacuous:xlate_checks_0", line)
