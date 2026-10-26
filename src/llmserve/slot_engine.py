"""Static-slot engine: preallocated KV cache, fixed-shape decode, optional CUDA graphs,
chunked prefill and prefix caching.

Why a second engine?  The reference ``Engine`` keeps a dynamic, left-padded KV cache whose shape changes
every step, so each decode step launches ~hundreds of small kernels from Python and pays the allocator/concat
cost of growing caches. On a T4 that makes decode ~30 ms/step at *any* batch size (about 10% of memory
bandwidth). ``SlotEngine`` removes that overhead:

  * KV cache  K/V[layer]: [slots, kv_heads, max_ctx, head_dim], allocated once. A request owns one slot for its life.
  * Decode runs a hand-written transformer step over the HF model's own weights/modules with fixed shapes:
    tokens [S], positions [S] (static buffers). Attention reads a bucketed prefix [:T] of the cache with a
    per-slot validity mask. GQA is handled without copying K/V (queries of one KV group form the query axis).
  * With ``cuda_graphs=True`` every (batch bucket, ctx bucket) shape is captured once as a CUDA graph and replayed.
  * Prefill still uses the HF model (batch of 1) so any architecture HF supports can prefill; its KV is copied into the slot.
  * ``chunk_size`` splits long prefills into pieces interleaved with decode steps (bounds decode stalls).
  * ``prefix_cache`` reuses KV of previously seen prompt prefixes (block-aligned, LRU, byte-capped).

Everything is greedy decoding, like the reference engine, so outputs are directly comparable.
"""
from __future__ import annotations

import collections
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from .engine import Engine, GenRequest, cache_to_tuples, forward_last_logits, tuples_to_cache


class PrefixCache:
    """Block-aligned prompt-prefix KV cache (LRU, byte capped). Stores one node per prompt; every block
    boundary of that prompt is indexed and points at views of the same tensors (no duplicate memory)."""

    def __init__(self, block=16, max_bytes=512 << 20):
        self.block, self.max_bytes = block, max_bytes
        self.nodes = collections.OrderedDict()  # id -> (keys, kv, nbytes)
        self.index = {}                          # tuple(prefix ids) -> node id
        self.bytes = 0
        self._next = 0
        self.lookups = self.hit_requests = self.hit_tokens = self.prompt_tokens = 0
        self.evictions = self.rejected_inserts = 0

    def lookup(self, ids):
        """Longest cached block-aligned prefix of `ids`, leaving at least one token to compute."""
        self.lookups += 1
        self.prompt_tokens += len(ids)
        m = (len(ids) - 1) // self.block
        while m >= 1:
            t = m * self.block
            nid = self.index.get(tuple(ids[:t]))
            if nid is not None and nid in self.nodes:
                _, kv, _ = self.nodes[nid]
                self.nodes.move_to_end(nid)
                self.hit_requests += 1
                self.hit_tokens += t
                return t, [(k[:, :, :t], v[:, :, :t]) for k, v in kv]
            m -= 1
        return 0, None

    def insert(self, ids, kv):
        m = len(ids) // self.block
        if m < 1:
            return
        top = tuple(ids[:m * self.block])
        if top in self.index and self.index[top] in self.nodes:
            return  # this prefix is already cached
        keys = [tuple(ids[:j * self.block]) for j in range(1, m + 1)]
        nbytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
        if nbytes > self.max_bytes:
            self.rejected_inserts += 1
            return
        nid = self._next
        self._next += 1
        self.nodes[nid] = (keys, kv, nbytes)
        self.bytes += nbytes
        for key in keys:
            self.index[key] = nid
        while self.bytes > self.max_bytes and len(self.nodes) > 1:
            old, (okeys, _, onb) = next(iter(self.nodes.items()))
            del self.nodes[old]
            self.evictions += 1
            self.bytes -= onb
            for key in okeys:
                if self.index.get(key) == old:
                    del self.index[key]

    def stats(self):
        return {"prefix_lookups": self.lookups, "prefix_hit_requests": self.hit_requests,
                "prefix_hit_tokens": self.hit_tokens, "prefix_prompt_tokens": self.prompt_tokens,
                "prefix_token_hit_rate": (self.hit_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0,
                "prefix_cache_mb": self.bytes / 2**20, "prefix_evictions": self.evictions,
                "prefix_rejected_inserts": self.rejected_inserts}


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _pow2_buckets(lo, hi):
    out, v = [], lo
    while v < hi:
        out.append(v)
        v *= 2
    out.append(hi)
    return sorted(set(b for b in out if b <= hi))


class SlotEngine(Engine):
    def __init__(self, model, eos_ids=(), device="cpu", max_batch=8, policy="continuous",
                 max_prefill_per_step=None, max_ctx=1024, log_steps=True, last_logits_only=True,
                 cuda_graphs=False, chunk_size=0, prefix_cache=False, prefix_block=16,
                 prefix_cache_mb=512, ctx_buckets=None, batch_buckets=None, **engine_kw):
        assert policy in ("continuous", "static")
        assert not (policy == "static" and chunk_size), "chunked prefill is only supported with continuous batching"
        if max_prefill_per_step is None:
            max_prefill_per_step = 1 if chunk_size else 2
        super().__init__(model, eos_ids=eos_ids, device=device, max_batch=max_batch, policy=policy,
                         max_prefill_per_step=max_prefill_per_step, max_ctx=max_ctx, log_steps=log_steps,
                         last_logits_only=last_logits_only, **engine_kw)
        cfg = model.config
        self.base = model.model
        self.nl = cfg.num_hidden_layers
        self.H = cfg.num_attention_heads
        self.KVH = getattr(cfg, "num_key_value_heads", None) or self.H
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.rep = self.H // self.KVH
        self.S = max_batch
        self.chunk_size = chunk_size
        self.cuda_graphs = bool(cuda_graphs) and str(device).startswith("cuda")
        self.graph_error: Optional[str] = None
        dtype = self.base.embed_tokens.weight.dtype
        self.dtype = dtype

        self.K = [torch.zeros(self.S, self.KVH, max_ctx, self.D, dtype=dtype, device=device) for _ in range(self.nl)]
        self.V = [torch.zeros_like(self.K[0]) for _ in range(self.nl)]
        pos = torch.arange(max_ctx, device=device)[None]
        cos, sin = self.base.rotary_emb(torch.zeros(1, dtype=dtype, device=device), pos)
        self.cos, self.sin = cos[0].to(dtype), sin[0].to(dtype)
        self.t_idx = torch.arange(max_ctx, device=device)
        self.ar = torch.arange(self.S, device=device)
        self.tok_buf = torch.zeros(self.S, dtype=torch.long, device=device)
        self.pos_buf = torch.zeros(self.S, dtype=torch.long, device=device)
        self.act_buf = torch.zeros(self.S, dtype=torch.long, device=device)

        self.ctx_buckets = sorted(set(ctx_buckets or _pow2_buckets(128, max_ctx)))
        self.batch_buckets = sorted(set(batch_buckets or _pow2_buckets(1, self.S)))
        self.free = list(range(self.S))
        self.slots = {}            # slot -> GenRequest
        self._pf = None            # prefill in progress
        self.slot_len = [0] * self.S
        self.pcache = PrefixCache(prefix_block, prefix_cache_mb << 20) if prefix_cache else None
        self.graphs = {}
        if self.cuda_graphs:
            try:
                self._capture_all()
            except Exception as exc:  # keep serving, eagerly
                self.graph_error = f"{type(exc).__name__}: {exc}"
                self.cuda_graphs = False
                self.graphs.clear()
                print(f"[slot-engine] CUDA graph capture failed, falling back to eager: {self.graph_error}", flush=True)

    # ------------------------------------------------------------------ fixed-shape decode
    @torch.no_grad()
    def _decode_core(self, Sb, T):
        """One decode step for slots [0, Sb) attending to cache positions [0, T). Returns next tokens [Sb]."""
        base, H, KVH, D, rep = self.base, self.H, self.KVH, self.D, self.rep
        tok, pos = self.tok_buf[:Sb], self.pos_buf[:Sb]
        ar = self.ar[:Sb]
        x = base.embed_tokens(tok)[:, None, :]
        cos = self.cos[pos][:, None, None, :]
        sin = self.sin[pos][:, None, None, :]
        mask = (self.t_idx[:T][None, :] <= pos[:, None])[:, None, None, :]
        for i, layer in enumerate(base.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(Sb, 1, H, D).transpose(1, 2)
            k = a.k_proj(h).view(Sb, 1, KVH, D).transpose(1, 2)
            v = a.v_proj(h).view(Sb, 1, KVH, D).transpose(1, 2)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            self.K[i][ar, :, pos] = k[:, :, 0]
            self.V[i][ar, :, pos] = v[:, :, 0]
            qg = q.reshape(Sb, KVH, rep, D)           # one KV head serves `rep` query heads: they form the query axis
            o = F.scaled_dot_product_attention(qg, self.K[i][:Sb, :, :T], self.V[i][:Sb, :, :T], attn_mask=mask)
            x = x + a.o_proj(o.reshape(Sb, 1, H * D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        logits = self.model.lm_head(base.norm(x)[:, 0])
        return logits.argmax(-1)

    def _capture_all(self):
        pool = torch.cuda.graph_pool_handle()
        for Sb in self.batch_buckets:
            for T in self.ctx_buckets:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._decode_core(Sb, T)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    out = self._decode_core(Sb, T)
                self.graphs[(Sb, T)] = (g, out)
        torch.cuda.synchronize()
        for t in self.K + self.V:   # warm-up runs wrote garbage into slot 0.. positions; start clean
            t.zero_()
        self.tok_buf.zero_(); self.pos_buf.zero_(); self.act_buf.zero_()

    @staticmethod
    def _bucket(value, buckets):
        for b in buckets:
            if b >= value:
                return b
        return buckets[-1]

    @torch.no_grad()
    def _decode_step(self):
        t0 = time.perf_counter()
        b = len(self.slots)
        Sb = self._bucket(max(self.slots) + 1, self.batch_buckets)
        T = self._bucket(max(self.slot_len[s] for s in self.slots) + 1, self.ctx_buckets)
        if self.cuda_graphs:
            g, out = self.graphs[(Sb, T)]
            g.replay()
        else:
            out = self._decode_core(Sb, T)
        toks = out.tolist()  # device sync
        self.tok_buf[:Sb].copy_(out)
        self.pos_buf.add_(self.act_buf)
        for s in sorted(self.slots):
            req = self.slots[s]
            self.slot_len[s] += 1
            if self._emit(req, toks[s]):
                self._release(s)
        if self.log_steps:
            self.step_log.append((t0, "decode", b, time.perf_counter() - t0, b))

    def _reap_active(self):
        now = time.perf_counter()
        pf = self._pf
        if pf is not None:
            r = pf["req"]
            why = "cancelled" if r.cancelled else "timeout" if self._expired(r, now) else None
            if why:
                self.cancelled_count += why == "cancelled"
                self.expired_count += why == "timeout"
                self._pf = None
                self._finish_unscheduled(r, why)
        for r, why in self._reap_active_reasons():
            for s, req in list(self.slots.items()):
                if req is r:
                    self.cancelled_count += why == "cancelled"
                    self.expired_count += why == "timeout"
                    self._release(s, count=False)
                    self._finish_unscheduled(r, why)

    def _release(self, s, count=True):
        del self.slots[s]
        self.act_buf[s] = 0
        self.pos_buf[s] = 0
        self.tok_buf[s] = 0
        self.slot_len[s] = 0
        self.free.append(s)
        self.free.sort()
        self.active = list(self.slots.values())
        self.completed += bool(count)

    # ------------------------------------------------------------------ prefill (HF model, batch of 1)
    def _has_work(self):
        return bool(self.pending or self.slots or self._pf is not None)

    def tick(self) -> bool:
        self._reap_pending()
        self._reap_active()
        started_empty = not self.slots and self._pf is None
        done_chunks = 0
        limit = self.max_batch if self.policy == "static" else self.max_prefill_per_step
        while done_chunks < limit:
            if self._pf is None:
                if not self.free:
                    break
                if self.policy == "static" and not started_empty:
                    break
                with self._cv:
                    if not self.pending:
                        break
                    req = self.pending.popleft()
                self._start_prefill(req)
            self._prefill_chunk()
            done_chunks += 1
        if self.slots:
            self._decode_step()
        with self._cv:
            return bool(self.pending or self.slots or self._pf is not None)

    def _start_prefill(self, req: GenRequest):
        cache, done = None, 0
        if self.pcache is not None:
            done, kv = self.pcache.lookup(req.prompt_ids)
            if done:
                cache = tuples_to_cache(kv)
        self._pf = {"req": req, "cache": cache, "done": done}

    @torch.no_grad()
    def _prefill_chunk(self):
        pf = self._pf
        req, ids = pf["req"], pf["req"].prompt_ids
        n, start = len(ids), pf["done"]
        end = n if self.chunk_size <= 0 else min(n, start + self.chunk_size)
        t0 = time.perf_counter()
        chunk = torch.tensor([ids[start:end]], device=self.device)
        kw = dict(use_cache=True)
        if pf["cache"] is not None:
            kw["past_key_values"] = pf["cache"]
        out = forward_last_logits(self.model, chunk, **kw) if self.last_logits_only else self.model(input_ids=chunk, **kw)
        pf["cache"], pf["done"] = out.past_key_values, end
        if end < n:
            if self.log_steps:
                self.step_log.append((t0, "prefill", 1, time.perf_counter() - t0, end - start))
            return
        tok = int(out.logits[0, -1].argmax())
        finished = self._emit(req, tok)
        if self.log_steps:
            self.step_log.append((t0, "prefill", 1, time.perf_counter() - t0, end - start))
        self._pf = None
        kv = cache_to_tuples(pf["cache"])
        if self.pcache is not None:
            self.pcache.insert(ids, kv)
        if finished:
            self.completed += 1
            return
        s = self.free.pop(0)
        for i, (k, v) in enumerate(kv):
            self.K[i][s, :, :n] = k[0]
            self.V[i][s, :, :n] = v[0]
        self.slots[s] = req
        self.slot_len[s] = n
        self.tok_buf[s] = tok
        self.pos_buf[s] = n
        self.act_buf[s] = 1
        self.active = list(self.slots.values())

    def stats(self):
        out = super().stats()
        out.update(engine="slot", cuda_graphs=self.cuda_graphs, graph_error=self.graph_error,
                   chunk_size=self.chunk_size, prefix_cache=self.pcache is not None,
                   n_graphs=len(self.graphs))
        if self.pcache is not None:
            out.update(self.pcache.stats())
        return out
