"""Build reports/SUPPORTDRAFT.md from the CSV/JSONL files written by compare.py.

Every number carries a label: MEASURED (observed in the run), AUTOMATED (deterministic quality check, not human-verified),
ASSUMPTION (an input you set), ESTIMATE (arithmetic on measured numbers and assumptions). Nothing here is a measured saving.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .cost import saturated_cost, scenario, sustainable_rate
from .baselines import score_baselines
from .retrieval import retrieval_recall
from .tickets import split

PROC = Path("results/processed")
RAW = Path("results/supportdraft")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"

DEFAULT_ASSUMPTIONS = dict(
    gpu_usd_per_hour=0.35,            # ASSUMED hourly price of a T4-class cloud GPU; set your own
    agents=50, drafts_per_agent_per_hour=6, service_hours_per_day=10,
    agent_cost_usd_per_hour=30.0,     # ASSUMED loaded cost of a support agent
    minutes_saved_sensitivity=[1, 3, 5], acceptance_sensitivity=[0.5, 0.7, 0.9],
    tickets_per_month_sensitivity=[2000, 10000])


DISPLAY = {"useful": "passing & on time", "useful drafts/s": "passing & on-time drafts/s", "useful_rate_at_cap": "passing_on_time_rate_at_cap",
           "saturated_usd_per_1k_useful": "saturated_usd_per_1k_passing", "est_provisioned_usd_per_1k_useful": "est_provisioned_usd_per_1k_passing"}


def _md(df, digits=2):
    if df.empty:
        return "(no rows)\n"
    cols = [DISPLAY.get(c, c) for c in df.columns]
    fmt = lambda v: (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _pct(x):
    return "n/a" if x != x else f"{100 * x:.0f}%"


SUMMARY = {}


def summary_markdown():
    """Compact, data-driven summary for the top-level README (call after build())."""
    if not SUMMARY:
        return ""
    setup, A, split_name = SUMMARY["setup"]
    q = SUMMARY["quality"][["backend", "overall correct", "answerable: correct", "over-escalated", "escalation recall", "invented numbers"]]
    return ("**SupportDraft** is an application built on this serving stack: a grounded support-reply drafter over a fictional product's "
            "knowledge base (30 passages, 114 synthetic tickets, held-out test split). All data are synthetic; automated checks are rule-based "
            "and not human-verified; dollar figures use an assumed GPU price of "
            f"{A['gpu_usd_per_hour']:.2f} USD/hour; no labor saving was measured.\n\n"
            f"Model `{setup.get('model', '?')}`, split `{split_name}`, draft must complete within {setup.get('e2e_slo_s', '?')} s to count as a passing, on-time draft.\n\n"
            "Draft quality (AUTOMATED), including two no-LLM baselines:\n\n" + _md(q) +
            "\nServing capacity and cost (MEASURED duration x ASSUMED price):\n\n" + _md(SUMMARY["capacity"], 3) +
            "\nFull report with caveats: [reports/SUPPORTDRAFT.md](reports/SUPPORTDRAFT.md).\n")


def load_assumptions():
    path = PROC / "supportdraft_assumptions.json"
    a = dict(DEFAULT_ASSUMPTIONS)
    if path.exists():
        a.update(json.loads(path.read_text()))
    return a


def build(assumptions=None, split_name="test"):
    """Report on one split (default: the held-out test split). Dev-split rows, if present, are ignored."""
    serving = pd.read_csv(PROC / "supportdraft_serving.csv")
    quality = pd.read_csv(PROC / "supportdraft_quality.csv")
    serving, quality = serving[serving["split"] == split_name], quality[quality["split"] == split_name]
    if serving.empty:
        raise SystemExit(f"no results for split '{split_name}'; run compare_backends(..., split_name='{split_name}') first")
    sp = PROC / f"supportdraft_setup_{split_name}.json"
    setup = json.loads(sp.read_text()) if sp.exists() else {}
    A = assumptions or load_assumptions()
    backends = list(dict.fromkeys(serving["backend"]))
    out = ["# SupportDraft: results\n"]
    out.append("All tickets and the knowledge base are **synthetic** (a fictional product, 'PebbleBoard'). Labels used below: "
               "**MEASURED** = observed in the run; **AUTOMATED** = deterministic quality check, not human-verified; "
               "**ASSUMPTION** = an input chosen by the author; **ESTIMATE** = arithmetic on measured numbers and assumptions. "
               "No reviewer time, user behaviour or real cost saving was measured.\n")
    out.append(f"Setup: model `{setup.get('model', '?')}`, split `{split_name}`, latency objective: draft complete within "
               f"{setup.get('e2e_slo_s', '?')} s (a draft counts as *passing and on time* only if it passes the automated checks AND arrives in time; this is not a human judgment of usefulness); "
               f"time to first token objective {setup.get('ttft_slo_s', '?')} s is reported separately.\n")

    # -------- retrieval
    out.append("## 1. Retrieval (MEASURED, backend-independent)\n")
    rr = pd.DataFrame([dict(split=s, **{f"recall@{k}": retrieval_recall(split(s), k) for k in (1, 3)}) for s in ("dev", "test")])
    out.append(_md(rr, 3))
    out.append("Share of answerable tickets whose correct passage is among the top-k retrieved. BM25 with light stemming over 30 "
               "passages is an easy retrieval problem; this does not predict retrieval on a large, messy knowledge base.\n")

    # -------- quality
    out.append("## 2. Draft quality (AUTOMATED, test split)\n")
    low = quality.sort_values("offered_rps").groupby("backend").head(1).set_index("backend").loc[backends]
    cols = [("correct_rate", "overall correct"), ("supported_answer_rate", "answerable: correct"), ("fact_coverage", "fact coverage"),
            ("citation_support_rate", "citation supports claim"), ("unsupported_number_rate", "invented numbers"),
            ("prohibited_claim_rate", "known-wrong values"), ("copy_share", "copied from passage"), ("over_escalation_rate", "over-escalated"), ("escalation_recall", "escalation recall")]
    base = score_baselines(split(split_name))
    qt = pd.DataFrame({"backend": list(low.index) + list(base),
                       **{lab: [_pct(v) for v in low[c]] + [_pct(b[c]) for b in base.values()] for c, lab in cols}})
    out.append(_md(qt))
    spread = quality.groupby("backend")["correct_rate"].agg(["min", "max"])
    out.append("Measured at each backend's lowest offered rate. Greedy decoding makes quality nearly independent of load; the spread "
               "of the overall correct rate across offered rates was " + ", ".join(f"{b}: {_pct(r['min'])}-{_pct(r['max'])}" for b, r in spread.iterrows())
               + ". The two baseline rows use no language model and are scored by the same checks: quoting the top retrieved passage "
               "already satisfies the rules on most answerable tickets, so the checks cannot separate a tailored reply from a copy-paste "
               "('copied from passage' shows how much of each draft is verbatim source text). A language model adds value mainly through "
               "escalation recall and through answering the actual question, which these checks only partly capture. These are rule-based "
               "checks: they verify required facts, cited-source support, invented numbers, known-wrong values and "
               "escalation. They do not judge tone, completeness or subtle errors; use the human review sheet for that.\n")
    # failure breakdown
    for b in backends:
        f = RAW / f"{b}_{split_name}_rate{float(low.loc[b, 'offered_rps']):g}.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            c = collections.Counter((r.get("score") or {}).get("failure") or "ok" for r in recs)
            out.append(f"- `{b}` outcomes: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    out.append("")
    kinds = collections.defaultdict(lambda: [0, 0])
    for b in backends:
        f = RAW / f"{b}_{split_name}_rate{float(low.loc[b, 'offered_rps']):g}.jsonl"
        if f.exists() and b == backends[0]:
            for l in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                kinds[r["kind"]][0] += int(r.get("correct", False)); kinds[r["kind"]][1] += 1
    if kinds:
        out.append(f"Correct rate by ticket kind for `{backends[0]}`: " + ", ".join(f"{k} {_pct(v[0]/v[1])} (n={v[1]})" for k, v in sorted(kinds.items())) + "\n")

    # -------- serving
    out.append("## 3. Serving under a ticket spike (MEASURED)\n")
    st = serving.copy()
    tab = st[["backend", "offered_rps", "n_requests", "failed", "ttft_p95", "e2e_p95", "on_time_rate", "useful_rate", "useful_per_s"]].copy()
    tab["on_time_rate"] = tab["on_time_rate"].map(_pct); tab["useful_rate"] = tab["useful_rate"].map(_pct)
    tab.columns = ["backend", "offered/s", "requests", "failed", "p95 TTFT (s)", "p95 draft time (s)", "on time", "useful", "useful drafts/s"]
    out.append(_md(tab))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for b in backends:
        g = st[st["backend"] == b].sort_values("offered_rps")
        ax[0].plot(g["offered_rps"], g["e2e_p95"], marker="o", label=b)
        ax[1].plot(g["offered_rps"], g["useful_per_s"], marker="o", label=b)
    ax[0].axhline(setup.get("e2e_slo_s", 15), color="k", ls=":", lw=1)
    ax[0].set_xlabel("offered tickets/s"); ax[0].set_ylabel("p95 draft time (s)"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[1].set_xlabel("offered tickets/s"); ax[1].set_ylabel("passing, on-time drafts/s"); ax[1].set_xscale("log")
    ax[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(FIGS / "supportdraft.png", dpi=150); plt.close()
    out.append("![supportdraft](figures/supportdraft.png)\n")

    # -------- capacity and cost
    out.append("## 4. Capacity and cost\n")
    out.append(f"ASSUMPTION: GPU price {A['gpu_usd_per_hour']:.2f} USD/hour (set `results/processed/supportdraft_assumptions.json` to change it). "
               "'Sustainable rate' = highest tested offered rate with zero failures and at least 95% of drafts on time.\n")
    rows = []
    for b in backends:
        g = st[st["backend"] == b]
        cap, up = sustainable_rate(g.to_dict("records"))
        row = dict(backend=b, sustainable_rate=("0 (never met)" if cap == 0 else f">= {cap:g} (never saturated)" if up is None else f"{cap:g} (next tested {up:g} failed)"))
        if cap > 0:
            r = g[g["offered_rps"] == cap].iloc[0]
            c = saturated_cost(r["useful"], r["wall_s"], A["gpu_usd_per_hour"])
            row.update(useful_rate_at_cap=_pct(r["useful_rate"]), saturated_usd_per_1k_useful=c["usd_per_1k_useful"])
            sc = scenario(cap, r["useful_rate"], A["gpu_usd_per_hour"], A["agents"], A["drafts_per_agent_per_hour"], A["service_hours_per_day"])
            row.update(est_agents_supported=sc["agents_supported"], est_provisioned_usd_per_1k_useful=sc["usd_per_1k_useful_provisioned"],
                       est_utilization=sc["utilization_of_sustainable"])
        rows.append(row)
    cap_tab = pd.DataFrame(rows)
    out.append(_md(cap_tab, 3))
    SUMMARY["quality"] = qt
    SUMMARY["capacity"] = cap_tab[[c for c in ("backend", "sustainable_rate", "useful_rate_at_cap", "saturated_usd_per_1k_useful") if c in cap_tab]]
    SUMMARY["setup"] = (setup, A, split_name)
    out.append(f"- *saturated_usd_per_1k_passing* (MEASURED duration x ASSUMED price): GPU cost while the benchmark ran, per 1,000 passing, on-time drafts; "
               "assumes the GPU is fully busy.\n"
               f"- *est_agents_supported*, *est_provisioned_usd_per_1k_passing*, *est_utilization* (ESTIMATE): assume {A['agents']} agents each "
               f"requesting {A['drafts_per_agent_per_hour']} drafts per hour for {A['service_hours_per_day']} service hours per day. The provisioned cost "
               "includes idle GPU time in the service window, so a lightly used deployment looks much more expensive per passing, on-time draft than the "
               "saturated figure suggests. Change the assumptions to match a real team.\n")

    # -------- value estimate
    out.append("## 5. Labor value (ESTIMATE only; no reviewer time was measured)\n")
    out.append("Monthly value = tickets per month x share of drafts a person accepts x minutes saved per accepted draft x agent cost per minute, minus GPU "
               f"cost. Agent cost is ASSUMED at {A['agent_cost_usd_per_hour']:.0f} USD/hour. Minutes saved and acceptance are **unknown until a pilot "
               "measures them**; the table is a sensitivity analysis, not a result.\n")
    ests = []
    gpu_month = A["gpu_usd_per_hour"] * A["service_hours_per_day"] * 22
    for tpm in A["tickets_per_month_sensitivity"]:
        for acc in A["acceptance_sensitivity"]:
            for mins in A["minutes_saved_sensitivity"]:
                ests.append(dict(tickets_per_month=tpm, acceptance=_pct(acc), minutes_saved_per_accepted_draft=mins,
                                 est_gross_value_usd=tpm * acc * mins * A["agent_cost_usd_per_hour"] / 60,
                                 est_gpu_cost_usd=gpu_month,
                                 est_net_usd=tpm * acc * mins * A["agent_cost_usd_per_hour"] / 60 - gpu_month))
    out.append(_md(pd.DataFrame(ests), 0))

    # -------- human review
    reviews = sorted(RAW.glob("review_*.csv")) if RAW.exists() else []
    if reviews:
        from .review import summarize_review
        out.append("## 6. Human review (PILOT-MEASURED, small sample)\n")
        rr_rows = [dict(sheet=p.name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize_review(p).items()}) for p in reviews]
        out.append(_md(pd.DataFrame(rr_rows), 3))

    out.append("## Limits\n")
    out.append("- Synthetic knowledge base and tickets written by the author; real tickets are messier. The test split was not used to tune prompts, "
               "but retrieval stemming was added after looking at retrieval misses that included test tickets.\n"
               "- Automated checks are rule-based; they can pass a poor draft and fail an acceptable one.\n"
               "- One model, one GPU, greedy decoding, Poisson arrivals; the ticket set is small (see `n` above).\n"
               "- Cost depends on an assumed GPU price; labor value depends on assumptions that were not measured.\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "SUPPORTDRAFT.md").write_text("\n".join(out), encoding="utf-8")
    return "\n".join(out)


if __name__ == "__main__":
    print(build())
