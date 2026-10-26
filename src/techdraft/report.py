"""Aggregate report for the TechQA workload. Writes reports/TECHDRAFT.md (aggregate numbers only; no licensed text)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from supportdraft.cost import sustainable_rate
from supportdraft.report import _md

from .data import KB, load, split_of
from . import manifest as _manifest
from . import run as _run
from .run import prepare_items
from .score import F1_PASS, score_baselines, tune_threshold

PROC = Path("results/processed")
REPORTS = Path("reports")
_pct = lambda v: f"{100 * v:.0f}%" if pd.notna(v) else "n/a"
SUMMARY = {}


def _quality_table(q, baselines=None):
    cols = ["pass_rate", "answerable_pass", "impossible_pass", "mean_f1_all_answerable", "evidence_recall", "grounded_rate",
            "abstain_precision", "abstain_recall", "unparsed_rate"]
    rows = []
    for _, r in q.iterrows():
        rows.append([r["backend"]] + [r.get(c, float("nan")) for c in cols])
    for name, m in (baselines or {}).items():
        rows.append([name] + [m.get(c, float("nan")) for c in cols])
    df = pd.DataFrame(rows, columns=["system", "pass", "answerable: pass", "unanswerable: abstained", "answer F1 (0 if abstained)", "evidence recall",
                                     "quotes grounded", "abstain precision", "abstain recall", "unparsed"])
    for c in df.columns[1:]:
        df[c] = df[c].map(_pct) if c != "answer F1 (0 if abstained)" else df[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    return df


def build(data_path=None, final="final", tune="tune", k=4):
    REPORTS.mkdir(exist_ok=True)
    records = load(data_path)
    kb = KB(records)
    it_tune = prepare_items([r for r in records if split_of(r["id"]) == tune], kb, k)
    ex_tune = list(_run.EXCLUDED)
    it_final = prepare_items([r for r in records if split_of(r["id"]) == final], kb, k)
    ex_final = list(_run.EXCLUDED)
    n_ex = len(ex_tune) + len(ex_final)
    man = _manifest.build(records, {tune: it_tune, final: it_final}, {tune: ex_tune, final: ex_final}, k)
    _manifest.write(man, PROC / "techdraft_manifest.json")
    thr = tune_threshold(it_tune, kb, k)
    base_final = score_baselines(it_final, kb, thr, k)
    base_tune = score_baselines(it_tune, kb, thr, k)
    n_imp = lambda its: sum(i["impossible"] for i in its)
    hit = lambda its: sum(any(kb.by_id[c]["doc"] in i["gold_docs"] for c, _ in i["retrieved"]) for i in its if not i["impossible"]) / max(1, len(its) - n_imp(its))
    srv = pd.read_csv(PROC / "techdraft_serving.csv")
    qual = pd.read_csv(PROC / "techdraft_quality.csv")
    out = ["# TechDraft: evidence-first technical-support answering on real forum questions\n"]
    out.append(
        "Data: the public TechQA questions (real IBM developer-forum posts) and the IBM technote documents they refer to, from the Hugging Face copy "
        "`nvidia/TechQA-RAG-Eval`. Raw data are **not** in this repository. Labels: **MEASURED** = observed in the run; **AUTOMATED** = rule-based overlap "
        "check, not a human judgment; no cost saving or productivity gain was measured.\n")
    out.append("## Setup and what this is not\n")
    out.append(
        f"- {len(records)} questions ({sum(r['is_impossible'] for r in records)} unanswerable) and {len(kb.docs)} documents ({len(kb.chunks)} chunks of about 180 words). "
        f"Split by hash of the question id: `{tune}` ({len(it_tune)} questions, used for prompt and threshold choices) and `{final}` ({len(it_final)}, evaluated after freezing).\n"
        "- This is **our own split of a merged copy**, not the official train/dev split. The knowledge base is built from the documents of *all* questions, so each "
        "question's correct document is always in the index, and unanswerable questions have only other questions' documents to match. That makes retrieval easier than a "
        "real deployment and makes abstention a test of whether the retrieved text answers the question.\n"
        f"- 'Pass' for an answerable question means: the model answered, every quoted sentence appears verbatim in the excerpt it cites, and the answer's token F1 against the gold answer is "
        f"at least {F1_PASS}. For an unanswerable question it means the model abstained. This is an overlap check, **not** a measure of whether a customer would find the reply useful.\n"
        f"- {n_ex} question(s) were excluded because the prompt would not fit the 4,096-token server limit with 300 generated tokens; they are not truncated or scored. Excluded ids: `{tune}` {ex_tune}, `{final}` {ex_final}. Every model and baseline below is computed on exactly the remaining ids; hashes of those id sets, the prompt and the parser are in `results/processed/techdraft_manifest.json`.\n"
        "- Unanswerable here means 'the provided documents do not contain the answer', not 'a human must act'.\n")
    out.append("## 1. Retrieval (MEASURED, backend-independent)\n")
    out.append(_md(pd.DataFrame([dict(split=tune, questions=len(it_tune), unanswerable=n_imp(it_tune), **{f"gold document in top-{k}": _pct(hit(it_tune))}),
                                 dict(split=final, questions=len(it_final), unanswerable=n_imp(it_final), **{f"gold document in top-{k}": _pct(hit(it_final))})])))
    MODELS = (("quality", "Qwen2.5-1.5B"), ("quality3b", "Qwen2.5-3B"))

    def _runs(split):
        parts = []
        for run, label in MODELS:
            q = qual[(qual["split"] == split) & (qual["run"] == run)]
            if q.empty:
                continue
            q = q.sort_values("offered_rps").groupby("backend", as_index=False).first()
            q["backend"] = label + " / " + q["backend"]
            parts.append(q)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    out.append(f"## 2. Answer quality (AUTOMATED, `{final}` split, one pass over every question)\n")
    q = _runs(final)
    if q.empty:
        out.append("(no quality run recorded for this split)\n")
    else:
        out.append(_md(_quality_table(q, base_final)))
        SUMMARY["quality"] = _quality_table(q, base_final)
    qt = _runs(tune)
    if not qt.empty:
        out.append(f"\nThe same measures on the `{tune}` split (the split used for choices):\n")
        out.append(_md(_quality_table(qt, base_tune)))
    out.append("\n## 3. Serving under load (MEASURED, `%s` split, sampled questions)\n" % final)
    sp = PROC / f"techdraft_setup_{final}_sweep.json"
    if sp.exists():
        out.append(f"Model served in the sweep: `{json.loads(sp.read_text()).get('model', '?')}`.\n")
    sw = srv[(srv["split"] == final) & (srv["run"] == "sweep")].sort_values(["backend", "offered_rps"])
    if sw.empty:
        out.append("(no sweep recorded)\n")
    else:
        tab = sw[["backend", "offered_rps", "n_requests", "failed", "ttft_p95", "e2e_p95", "on_time_rate", "useful_rate"]].copy()
        tab["on_time_rate"] = tab["on_time_rate"].map(_pct); tab["useful_rate"] = tab["useful_rate"].map(_pct)
        tab.columns = ["backend", "offered/s", "requests", "failed", "p95 TTFT (s)", "p95 draft time (s)", "on time", "passing & on time"]
        out.append(_md(tab))
        caps = []
        for b, g in sw.groupby("backend"):
            cap, nxt = sustainable_rate(g.to_dict("records"))
            caps.append(dict(backend=b, sustainable_rate=cap, next_tested_rate=nxt if nxt is not None else "not reached"))
        out.append("\nSustainable rate = highest tested offered rate with zero failures and at least 95% of drafts on time:\n")
        out.append(_md(pd.DataFrame(caps), 2))
        SUMMARY["capacity"] = pd.DataFrame(caps)
    out.append("\n## Limits\n")
    out.append(
        "- One model, one GPU, greedy decoding, Poisson arrivals; the evaluation set is a few hundred questions, so a few points of difference are noise.\n"
        "- The split is ours, on a merged copy; the final set was evaluated after freezing the pipeline, but any later tuning against it would remove that property.\n"
        "- Overlap with a gold answer rewards copying; the top-chunk baselines show how much of the score a no-LLM system gets.\n"
        "- The pass rule checks that quoted sentences appear in the excerpts and that the answer overlaps the gold answer; it does not check that every statement in the answer follows from the quotes.\n"
        "- No human judged these replies. Prompts are long (retrieved technote excerpts), so serving results differ from short-prompt workloads.\n")
    md = "\n".join(out)
    (REPORTS / "TECHDRAFT.md").write_text(md, encoding="utf-8")
    return md
