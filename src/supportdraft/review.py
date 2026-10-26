"""Human review of drafts: build a review sheet, collect approve/edit/reject verdicts with timing, and compare to the automated checks.

    python -m supportdraft.review make results/supportdraft/slot-graph-32_test_rate1.jsonl --n 30
    python -m supportdraft.review run  results/supportdraft/review_slot-graph-32.csv
    python -m supportdraft.review summary results/supportdraft/review_slot-graph-32.csv

Human verdicts are the only way to learn whether the automated checks agree with a person; the report shows that agreement
and never presents automated scores as human-verified.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from .kb import BY_ID

VERDICTS = {"a": "approve", "e": "edit", "r": "reject"}


def make_review_csv(jsonl, n=30, seed=0, out=None):
    recs = [json.loads(line) for line in Path(jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, uniq = set(), []
    for r in recs:
        if r["ok"] and r["ticket_id"] not in seen:
            seen.add(r["ticket_id"])
            uniq.append(r)
    random.Random(seed).shuffle(uniq)
    rows = [dict(ticket_id=r["ticket_id"], kind=r["kind"], question="", sources_shown=" ".join(r["retrieved_ids"]), draft=r["draft"],
                 automated_correct=bool(r["correct"]), human_verdict="", review_seconds="", notes="") for r in uniq[:n]]
    from .tickets import TICKETS
    q = {t["ticket_id"]: t["question"] for t in TICKETS}
    for row in rows:
        row["question"] = q[row["ticket_id"]]
    out = Path(out) if out else Path(jsonl).parent / ("review_" + Path(jsonl).stem.split("_")[0] + ".csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_interactive(csv_path):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for i, row in df.iterrows():
        if row["human_verdict"]:
            continue
        print("\n" + "=" * 78 + f"\nTicket {row['ticket_id']} ({row['kind']}):\n{row['question']}\n")
        for pid in row["sources_shown"].split():
            print(f"  [{pid}] {BY_ID[pid]['text'][:160]}...")
        print(f"\nDRAFT:\n{row['draft']}\n")
        t0 = time.time()
        while True:
            v = input("[a]pprove as is / [e]dit needed / [r]eject / [s]kip / [q]uit > ").strip().lower()
            if v in VERDICTS or v in ("s", "q"):
                break
        if v == "q":
            break
        if v == "s":
            continue
        df.loc[i, "human_verdict"] = VERDICTS[v]
        df.loc[i, "review_seconds"] = f"{time.time() - t0:.1f}"
        df.to_csv(csv_path, index=False)
    return df


def summarize_review(csv_path):
    df = pd.read_csv(csv_path)
    done = df[df["human_verdict"].isin(VERDICTS.values())].copy()
    if done.empty:
        return {"n_reviewed": 0}
    done["human_ok"] = done["human_verdict"].isin(["approve", "edit"])
    done["automated_correct"] = done["automated_correct"].astype(bool)
    agree = (done["human_ok"] == done["automated_correct"]).mean()
    return dict(n_reviewed=len(done), approve_rate=(done["human_verdict"] == "approve").mean(),
                accept_or_edit_rate=done["human_ok"].mean(), reject_rate=(done["human_verdict"] == "reject").mean(),
                automated_vs_human_agreement=agree,
                auto_pass_but_human_reject=int(((done["automated_correct"]) & (~done["human_ok"])).sum()),
                auto_fail_but_human_accept=int(((~done["automated_correct"]) & (done["human_ok"])).sum()),
                mean_review_seconds=pd.to_numeric(done["review_seconds"], errors="coerce").mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run", "summary"])
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "make":
        print("wrote", make_review_csv(a.path, a.n))
    elif a.cmd == "run":
        run_interactive(a.path)
    else:
        print(json.dumps(summarize_review(a.path), indent=2, default=float))
