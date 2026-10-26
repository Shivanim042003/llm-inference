"""TechQA-RAG-Eval adapter: download, deterministic split, chunking with offsets, and an inverted-index BM25 over all chunks."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from supportdraft.retrieval import tokenize

HF_ID = "nvidia/TechQA-RAG-Eval"
CACHE = Path("data/techdraft")
CHUNK_WORDS, OVERLAP = 180, 30


def download(path=None):
    """Save the Hugging Face copy as jsonl under data/ (git-ignored). Needs `pip install datasets` and network."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, split="train")
    path = Path(path or CACHE / "techqa_rag_eval.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({k: r[k] for k in ("id", "question", "answer", "is_impossible", "contexts")}, ensure_ascii=False) + "\n")
    return path


def load(path=None):
    path = Path(path or CACHE / "techqa_rag_eval.jsonl")
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["id"] = str(r["id"])
        r["answer"] = r.get("answer") or ""
        r["is_impossible"] = bool(r.get("is_impossible"))
        r["contexts"] = [c for c in (r.get("contexts") or []) if c.get("text")]
        recs.append(r)
    return recs


def split_of(rec_id, tune_share=0.4):
    """Deterministic split by hash of the id: 'tune' (for prompt/threshold choices) or 'final' (evaluate once after freezing)."""
    h = int(hashlib.sha1(str(rec_id).encode()).hexdigest(), 16) % 1000
    return "tune" if h < tune_share * 1000 else "final"


def chunk_doc(doc_id, text, words=CHUNK_WORDS, overlap=OVERLAP):
    """Word-window chunks that keep character offsets into the original text (chunk text is an exact slice)."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    out, i, step = [], 0, max(1, words - overlap)
    while i < len(spans):
        j = min(len(spans), i + words)
        s, e = spans[i][0], spans[j - 1][1]
        out.append(dict(doc=doc_id, start=s, end=e, text=text[s:e]))
        if j == len(spans):
            break
        i += step
    return out


class KB:
    """All distinct documents from every record, chunked and indexed (BM25 with an inverted index)."""

    def __init__(self, records, k1=1.5, b=0.75):
        docs = {}
        for r in records:
            for c in r["contexts"]:
                docs.setdefault(c["filename"], c["text"])
        self.docs = docs
        self.chunks, self.by_id = [], {}
        for fn in sorted(docs):
            for ch in chunk_doc(fn, docs[fn]):
                ch["id"] = f"C{len(self.chunks):05d}"
                self.chunks.append(ch)
                self.by_id[ch["id"]] = ch
        self.k1, self.b = k1, b
        self.tf, self.len, self.post = [], [], defaultdict(list)
        for i, ch in enumerate(self.chunks):
            toks = tokenize(ch["text"])
            tf = Counter(toks)
            self.tf.append(tf)
            self.len.append(len(toks))
            for w in tf:
                self.post[w].append(i)
        n = len(self.chunks)
        self.avg = (sum(self.len) / n) if n else 1.0
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.post.items()}

    def retrieve(self, question, k=4):
        """Top-k chunks as [(chunk_id, bm25_score)] (only chunks with a positive score)."""
        sc = defaultdict(float)
        for w in set(tokenize(question)):
            for i in self.post.get(w, ()):
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        top = sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.chunks[i]["id"], s) for i, s in top]


def gold_docs(rec):
    return {c["filename"] for c in rec["contexts"]}
