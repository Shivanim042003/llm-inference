"""No-LLM reference points, scored with the same checks: what retrieval alone (or giving up) achieves."""
from __future__ import annotations

from .kb import BY_ID
from .retrieval import retrieve
from .score import score_draft, summarize_quality


def always_escalate(ticket):
    return "ESCALATE: needs a person"


def top_passage(ticket, k=1):
    """Quote the top retrieved passage with a citation. Never escalates."""
    ids = retrieve(ticket["question"], k)
    return " ".join(f"{BY_ID[i]['text']} [{i}]" for i in ids) if ids else "ESCALATE: no matching passage"


BASELINES = {"baseline: always escalate": always_escalate, "baseline: quote top-1 passage (no LLM)": top_passage}


def score_baselines(tickets, k_retrieved=3):
    by_id = {t["ticket_id"]: t for t in tickets}
    out = {}
    for name, fn in BASELINES.items():
        scored = [(t["ticket_id"], score_draft(t, fn(t), retrieve(t["question"], k_retrieved))) for t in tickets]
        out[name] = summarize_quality(scored, by_id)
    return out
