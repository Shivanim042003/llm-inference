"""Iteration-level (continuous) batching engine for Hugging Face causal LMs.

Scheduling policies (same code path, so comparisons isolate the policy):
  * max_batch=1                      -> sequential serving (one request at a time)
  * policy="static",     max_batch=B -> a new batch is admitted only when the previous one fully drains
  * policy="continuous", max_batch=B -> requests join/leave the running batch every decode step

KV cache layout: left-padded batch [B, H, T, D] with an explicit attention mask and per-row
position ids, so rows of different lengths can share one forward pass. When a request is
admitted, its prefill runs separately (batch of 1) and its cache is left-padded and
concatenated into the running batch; finished rows are dropped and all-padding columns trimmed.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from transformers import DynamicCache


def cache_to_tuples(cache):
    """Extract [(K, V), ...] per layer from a DynamicCache across transformers versions."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def tuples_to_cache(tuples):
    cache = DynamicCache()
    for i, (k, v) in enumerate(tuples):
        cache.update(k, v, i)
    return cache


_LTK = {"kw": None}  # which "keep only the last logits" kwarg this transformers version supports


def forward_last_logits(model, input_ids, **kwargs):
    """Forward pass that only computes the lm_head for the final position.

    Prefill otherwise projects every prompt position onto the full vocabulary
    (prompt_len x hidden x vocab FLOPs, and a [B, L, vocab] logits tensor) only to discard all but the last row.
    """
    candidates = [_LTK["kw"]] if _LTK["kw"] else ["logits_to_keep", "num_logits_to_keep"]
    for kw in candidates:
        try:
            out = model(input_ids=input_ids, **{kw: 1}, **kwargs)
            _LTK["kw"] = kw
            return out
        except TypeError:
            continue
    _LTK["kw"] = None
    return model(input_ids=input_ids, **kwargs)


@dataclass
class GenRequest:
    prompt_ids: List[int]
    max_new_tokens: int
    ignore_eos: bool = False
    on_token: Optional[Callable[[int, bool], None]] = None
    # filled in by the engine
    deadline_s: Optional[float] = None      # seconds after submit; expired requests are removed from the scheduler
    cancelled: bool = False                 # set by Engine.cancel (e.g. client disconnected)
    t_submit: float = 0.0
    t_first: Optional[float] = None
    t_done: Optional[float] = None
    out_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None


class Overloaded(Exception):
    """Raised by Engine.submit when the pending queue or the token budget is full (maps to HTTP 503)."""


class EngineClosed(Exception):
    """Raised by Engine.submit after shutdown started or the engine loop failed."""


class Engine:
    def __init__(self, model, eos_ids=(), device="cpu", max_batch=8, policy="continuous",
                 max_prefill_per_step=2, max_ctx=4096, log_steps=True, last_logits_only=True,
                 max_pending=256, max_queued_tokens=None):
        assert policy in ("continuous", "static")
        self.model = model
        self.max_pending = max_pending
        self.max_queued_tokens = max_queued_tokens      # budget over prompt+max_new tokens of all unfinished requests
        self.accepting = True
        self._pf = None
        self.failed: Optional[str] = None
        self.rejected = self.cancelled_count = self.expired_count = 0
        self.eos_ids = set(eos_ids)
        self.device = device
        self.max_batch = max_batch
        self.policy = policy
        self.max_prefill_per_step = max_prefill_per_step
        self.max_ctx = max_ctx
        self.log_steps = log_steps
        self.last_logits_only = last_logits_only

        self.pending = collections.deque()
        self.active: List[GenRequest] = []
        self.cache = None
        self.mask = None
        self.next_ids = None
        self.step_log: List[tuple] = []  # (t, kind, batch_size, seconds, tokens_in_step)
        self.completed = 0

        self._cv = threading.Condition()
        self._stop = False
        self._thread = None

    # ------------------------------------------------------------------ public API
    def submit(self, req: GenRequest):
        """Queue a request. Raises Overloaded / EngineClosed instead of queueing without bound.

        Prompts that do not fit in max_ctx together with max_new_tokens are truncated here only for direct
        library use; the HTTP server rejects them with 400 before they reach the engine.
        """
        if self.failed is not None or not self.accepting:
            raise EngineClosed(self.failed or "engine is shutting down")
        if len(req.prompt_ids) + req.max_new_tokens > self.max_ctx:
            keep = max(1, self.max_ctx - req.max_new_tokens)
            req.prompt_ids = req.prompt_ids[-keep:]
        req.t_submit = time.perf_counter()
        with self._cv:
            if len(self.pending) >= self.max_pending:
                self.rejected += 1
                raise Overloaded(f"pending queue full ({self.max_pending})")
            if self.max_queued_tokens is not None:
                load = sum(len(r.prompt_ids) + r.max_new_tokens for r in list(self.pending) + list(self.active))
                if load + len(req.prompt_ids) + req.max_new_tokens > self.max_queued_tokens:
                    self.rejected += 1
                    raise Overloaded(f"token budget exhausted ({load}/{self.max_queued_tokens})")
            self.pending.append(req)
            self._cv.notify()

    def cancel(self, req: GenRequest):
        """Thread-safe: mark a request cancelled; the engine thread removes it at the next tick."""
        req.cancelled = True
        with self._cv:
            self._cv.notify()

    def _finish_unscheduled(self, req: GenRequest, reason: str):
        req.t_done, req.finish_reason = time.perf_counter(), reason
        if req.on_token:
            req.on_token(None, True)

    def _expired(self, req: GenRequest, now: float) -> bool:
        return req.deadline_s is not None and now - req.t_submit > req.deadline_s

    def _reap_pending(self):
        """Drop cancelled / expired requests that have not started (cheap: no KV state yet)."""
        now = time.perf_counter()
        with self._cv:
            keep, dead = collections.deque(), []
            for r in self.pending:
                if r.cancelled:
                    dead.append((r, "cancelled"))
                elif self._expired(r, now):
                    dead.append((r, "timeout"))
                else:
                    keep.append(r)
            self.pending = keep
        for r, why in dead:
            self.cancelled_count += why == "cancelled"
            self.expired_count += why == "timeout"
            self._finish_unscheduled(r, why)

    def _reap_active_reasons(self):
        """[(request, reason)] for active requests that must stop now."""
        now = time.perf_counter()
        out = []
        for r in self.active:
            if r.cancelled:
                out.append((r, "cancelled"))
            elif self._expired(r, now):
                out.append((r, "timeout"))
        return out

    def fail_all(self, reason: str):
        """Terminate every unfinished request with `reason` (engine failure / shutdown) so no client hangs."""
        with self._cv:
            victims = list(self.pending)
            self.pending.clear()
        victims += [r for r in self.active if r.t_done is None]
        pf = getattr(self, "_pf", None)
        if pf is not None:
            victims.append(pf["req"])
        for r in victims:
            if r.t_done is None:
                self._finish_unscheduled(r, reason)
        self.active = []
        if hasattr(self, "slots"):
            self.slots.clear()
        self._pf = None

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, drain_s: float = 0.0):
        """Stop accepting, let in-flight work finish for up to drain_s seconds, then fail whatever is left."""
        self.accepting = False
        t0 = time.perf_counter()
        while drain_s > 0 and self._has_work() and time.perf_counter() - t0 < drain_s and self._thread \
                and self._thread.is_alive():
            time.sleep(0.02)
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=10)
        self.fail_all("shutdown")

    def alive(self) -> bool:
        return self.failed is None and self._thread is not None and self._thread.is_alive()

    def run_until_idle(self):
        """Synchronous driver, used by tests."""
        while self.tick():
            pass

    def stats(self):
        return {"active": len(self.active), "pending": len(self.pending),
                "completed": self.completed, "max_batch": self.max_batch, "policy": self.policy,
                "last_logits_only": self.last_logits_only, "rejected": self.rejected,
                "cancelled": self.cancelled_count, "expired": self.expired_count,
                "failed": self.failed, "accepting": self.accepting}

    # ------------------------------------------------------------------ loop
    def _loop(self):
        while True:
            with self._cv:
                while not self._stop and not self._has_work():
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
            try:
                self.tick()
            except Exception as exc:  # propagate to every waiting client instead of dying silently
                self.failed = f"{type(exc).__name__}: {exc}"
                self.accepting = False
                self.fail_all("error")
                return

    def _has_work(self) -> bool:
        return bool(self.pending or self.active)

    def tick(self) -> bool:
        self._reap_pending()
        self._reap_active()
        started_empty = not self.active
        admitted = 0
        limit = self.max_batch if self.policy == "static" else self.max_prefill_per_step
        while len(self.active) < self.max_batch and admitted < limit:
            if self.policy == "static" and not started_empty:
                break
            with self._cv:
                if not self.pending:
                    break
                req = self.pending.popleft()
            self._admit(req)
            admitted += 1
        if self.active:
            self._decode_step()
        with self._cv:
            return bool(self.active or self.pending)

    # ------------------------------------------------------------------ internals
    @torch.no_grad()
    def _admit(self, req: GenRequest):
        t0 = time.perf_counter()
        ids = torch.tensor([req.prompt_ids], device=self.device)
        if self.last_logits_only:
            out = forward_last_logits(self.model, ids, use_cache=True)
        else:
            out = self.model(input_ids=ids, use_cache=True)
        tok = int(out.logits[0, -1].argmax())
        finished = self._emit(req, tok)
        if self.log_steps:
            self.step_log.append((t0, "prefill", 1, time.perf_counter() - t0, len(req.prompt_ids)))
        if finished:
            self.completed += 1
            return
        kv_new = cache_to_tuples(out.past_key_values)
        mask_new = torch.ones(1, ids.shape[1], dtype=torch.long, device=self.device)
        tok_t = torch.tensor([[tok]], device=self.device)
        if not self.active:
            self.cache = tuples_to_cache(kv_new)
            self.mask = mask_new
            self.next_ids = tok_t
        else:
            kv_old = cache_to_tuples(self.cache)
            t_old, t_new = self.mask.shape[1], mask_new.shape[1]
            t = max(t_old, t_new)
            merged = []
            for (ko, vo), (kn, vn) in zip(kv_old, kv_new):
                ko, vo = F.pad(ko, (0, 0, t - t_old, 0)), F.pad(vo, (0, 0, t - t_old, 0))
                kn, vn = F.pad(kn, (0, 0, t - t_new, 0)), F.pad(vn, (0, 0, t - t_new, 0))
                merged.append((torch.cat([ko, kn], 0), torch.cat([vo, vn], 0)))
            self.cache = tuples_to_cache(merged)
            self.mask = torch.cat([F.pad(self.mask, (t - t_old, 0)), F.pad(mask_new, (t - t_new, 0))], 0)
            self.next_ids = torch.cat([self.next_ids, tok_t], 0)
        self.active.append(req)

    def _reap_active(self):
        dead = self._reap_active_reasons()
        if not dead or not self.active:
            return
        ids = {id(r): why for r, why in dead}
        keep = [i for i, r in enumerate(self.active) if id(r) not in ids]
        for r, why in dead:
            self.cancelled_count += why == "cancelled"
            self.expired_count += why == "timeout"
            self._finish_unscheduled(r, why)
        self._drop_rows(keep)

    def _drop_rows(self, keep):
        """Keep only batch rows `keep`: free their KV, trim leading all-padding columns."""
        self.active = [self.active[i] for i in keep]
        if not keep:
            self.cache = self.mask = self.next_ids = None
            return
        idx = torch.tensor(keep, device=self.device)
        kv = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache_to_tuples(self.cache)]
        mask = self.mask.index_select(0, idx)
        first = int(mask.any(0).nonzero()[0])
        if first > 0:
            kv = [(k[:, :, first:], v[:, :, first:]) for k, v in kv]
            mask = mask[:, first:]
        self.cache = tuples_to_cache(kv)
        self.mask = mask
        self.next_ids = self.next_ids.index_select(0, idx)

    @torch.no_grad()
    def _decode_step(self):
        t0 = time.perf_counter()
        b = len(self.active)
        pos = self.mask.sum(1, keepdim=True)
        attn = torch.cat([self.mask, torch.ones(b, 1, dtype=self.mask.dtype, device=self.device)], 1)
        out = self.model(input_ids=self.next_ids, attention_mask=attn, position_ids=pos,
                         past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.mask = attn
        toks = out.logits[:, -1, :].argmax(-1)
        tok_list = toks.tolist()  # device sync
        keep = []
        for i, (req, tok) in enumerate(zip(self.active, tok_list)):
            if self._emit(req, tok):
                self.completed += 1
            else:
                keep.append(i)
        if self.log_steps:
            self.step_log.append((t0, "decode", b, time.perf_counter() - t0, b))
        if len(keep) == b:
            self.next_ids = toks.unsqueeze(1)
            return
        self.active = [self.active[i] for i in keep]
        if not keep:
            self.cache = self.mask = self.next_ids = None
            return
        idx = torch.tensor(keep, device=self.device)
        kv = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache_to_tuples(self.cache)]
        mask = self.mask.index_select(0, idx)
        first = int(mask.any(0).nonzero()[0])  # drop leading all-padding columns
        if first > 0:
            kv = [(k[:, :, first:], v[:, :, first:]) for k, v in kv]
            mask = mask[:, first:]
        self.cache = tuples_to_cache(kv)
        self.mask = mask
        self.next_ids = toks.index_select(0, idx).unsqueeze(1)

    def _emit(self, req: GenRequest, tok: int) -> bool:
        now = time.perf_counter()
        if req.t_first is None:
            req.t_first = now
        finished, reason = False, None
        is_eos = (not req.ignore_eos) and tok in self.eos_ids
        if not is_eos:
            req.out_ids.append(tok)
        if is_eos:
            finished, reason = True, "stop"
        elif len(req.out_ids) >= req.max_new_tokens:
            finished, reason = True, "length"
        if finished:
            req.t_done, req.finish_reason = now, reason
        if req.on_token:
            req.on_token(None if is_eos else tok, finished)   # None = terminal event with no text
        return finished
