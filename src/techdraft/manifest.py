"""Evaluation manifest: exactly which questions, prompt, parser and settings produced the reported numbers (no licensed text)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import pipeline, run, score
from .data import CHUNK_WORDS, HF_ID, OVERLAP

PROC = Path("results/processed")


def _h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _ids_hash(ids):
    return _h("\n".join(sorted(ids)))


def build(records, items_by_split, excluded_by_split, k=4, models=None):
    """items_by_split: {'tune': [...], 'final': [...]} as returned by run.prepare_items; excluded_by_split: ids dropped for length."""
    splits = {}
    for name, items in items_by_split.items():
        ids = [i["id"] for i in items]
        splits[name] = dict(n=len(ids), n_unanswerable=sum(i["impossible"] for i in items), ids_sha256_16=_ids_hash(ids),
                            excluded_ids=sorted(excluded_by_split.get(name, [])), excluded_sha256_16=_ids_hash(excluded_by_split.get(name, [])))
    return dict(
        dataset=HF_ID, n_records=len(records), split_rule="sha1(id) mod 1000 < 400 -> tune, else final",
        chunk_words=CHUNK_WORDS, chunk_overlap=OVERLAP, k_chunks=k,
        exclusion_rule=f"prompt tokens (Qwen chat template) > {run.MAX_PROMPT_TOKENS} (server limit 4096 minus 300 generated tokens)",
        tokenizer=run.TOKENIZER_ID, f1_pass_threshold=score.F1_PASS,
        system_prompt_sha256_16=_h(pipeline.SYSTEM), pipeline_py_sha256_16=_h(Path(pipeline.__file__).read_text(encoding="utf-8").strip()),
        splits=splits, models=models or ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
        note="All models and baselines in the report are computed on exactly the evaluated ids of each split.")


def write(manifest, path=None):
    path = Path(path or PROC / "techdraft_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
