"""Dependency-free BM25 retrieval over the knowledge base, and the grounded prompt."""
from __future__ import annotations

import math
import re
from collections import Counter

from .kb import BY_ID, KB, PRODUCT

STOP = set("a an and are as at be but by can do does for from has have how i if in is it my of on or our so that the "
           "their there this to was we what when where which who why will with you your me us not no".split())


def stem(w):
    """Very light suffix stripping (a general fix for word forms: invited / invitation / invite)."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9$%]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.ids = [d["id"] for d in docs]
        self.toks = [tokenize(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]   # title counted twice
        self.k1, self.b = k1, b
        self.avg = sum(len(t) for t in self.toks) / len(self.toks)
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, toks in zip(self.tf, self.toks):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg))
            out.append(s)
        return out

    def top(self, query, k=3):
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]
        return [self.ids[i] for i in order if sc[i] > 0]


_INDEX = None


def retrieve(question, k=3):
    global _INDEX
    if _INDEX is None:
        _INDEX = BM25(KB)
    return _INDEX.top(question, k)


SYSTEM = (f"You draft replies for {PRODUCT} support agents. A human agent reviews every draft before it is sent.\n"
          "Rules:\n"
          "1. Use ONLY the numbered passages provided. Never use outside knowledge and never invent prices, limits, dates or policies.\n"
          "2. Every sentence that states a fact MUST end with the id of the passage it came from in square brackets, for example [P07]. "
          "A reply with no citation is rejected.\n"
          "3. If the passages do not clearly contain the answer, or the request needs a person (account access or security problems, legal "
          "requests, billing disputes, custom quotes, bug reports, feature requests, compliance questions, anything about a specific account), "
          "reply with exactly one line: ESCALATE: <short reason>. Do not answer from general knowledge and do not guess.\n"
          "4. Be polite and direct, and keep the reply under 120 words.\n\n"
          "Example A (different product)\n"
          "Passage [P90] Widgets: Widgets can be renamed in Settings > General. Names can have up to 40 characters.\n"
          "Ticket: How do I rename a widget?\n"
          "Draft reply: You can rename a widget in Settings > General [P90]. Names can have up to 40 characters [P90].\n\n"
          "Example B (different product)\n"
          "Passage [P90] Widgets: Widgets can be renamed in Settings > General.\n"
          "Ticket: Can you add a dark theme to widgets?\n"
          "Draft reply: ESCALATE: feature request, not covered by the passages")


def build_messages(question, passage_ids):
    block = "\n".join(f"[{pid}] {BY_ID[pid]['title']}: {BY_ID[pid]['text']}" for pid in passage_ids) or "(no passages found)"
    user = f"Passages:\n{block}\n\nCustomer ticket:\n{question}\n\nDraft reply:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def retrieval_recall(tickets, k=3):
    """Share of answerable tickets whose acceptable source is in the top-k (retrieval quality, backend-independent)."""
    ans = [t for t in tickets if not t["must_escalate"]]
    hit = sum(any(s in retrieve(t["question"], k) for s in t["acceptable_source_ids"]) for t in ans)
    return hit / len(ans) if ans else float("nan")
