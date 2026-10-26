"""Deterministic (automated) quality checks for a drafted reply. These are NOT human-verified quality scores.

Checks, scored separately:
  citation_valid    every cited id exists in the knowledge base
  has_gold_citation at least one cited id is an acceptable source for the ticket
  citation_support  every required fact appears in the TEXT of the cited passages (a valid id alone is not enough)
  fact_coverage     share of required fact groups stated in the draft
  unsupported_nums  numbers/prices in the draft that appear in neither the cited passages nor the ticket
  prohibited_hit    the draft contains a known-wrong value for this topic
  too_long          more than MAX_WORDS words (the prompt asks for under 120; dumping the knowledge base is not an answer)
  cited_unseen      cites a passage that was not in the retrieved context given to the model
  escalation        ESCALATE reply on tickets that must be escalated (and not on ones that must not)
"""
from __future__ import annotations

import re

from .kb import BY_ID

MAX_WORDS = 160
NUM = re.compile(r"(?<![A-Za-z0-9])\$?\d[\d,]*(?:\.\d+)?%?")
CITE = re.compile(r"\bP\d{2}\b")


def _norm(s):
    s = s.lower().replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    return re.sub(r"\s+", " ", s)


def copy_share(draft, source_text, n=5):
    """Share of the draft's word n-grams that appear verbatim in the cited passages (1.0 = pure copy-paste). Informational only."""
    w = re.findall(r"[a-z0-9$%']+", _norm(draft))
    src = re.findall(r"[a-z0-9$%']+", _norm(source_text))
    if len(w) < n:
        return 0.0
    grams = {tuple(src[i:i + n]) for i in range(len(src) - n + 1)}
    mine = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return sum(g in grams for g in mine) / len(mine)


def is_escalation(draft):
    return _norm(draft).strip(" \t\n\r*_>`\"'").startswith("escalate")


def _numbers(text):
    out = set()
    for m in NUM.finditer(text):
        tok = m.group(0).rstrip(",").replace(",", "")
        end = m.end()
        if re.fullmatch(r"\d", tok) and text[end:end + 1] in (".", ")"):   # list marker like "1." or "2)"
            continue
        out.add(tok)
    return out


def score_draft(ticket, draft, retrieved_ids=None):
    d = _norm(draft)
    esc = is_escalation(draft)
    cited = list(dict.fromkeys(CITE.findall(draft)))
    valid = all(c in BY_ID for c in cited)
    cited_text = _norm(" ".join(BY_ID[c]["text"] for c in cited if c in BY_ID))
    groups = ticket["required_facts"]
    stated = [any(a in d for a in g) for g in groups]
    supported = [any(a in cited_text for a in g) for g in groups]
    nums = _numbers(draft) - _numbers(" ".join(BY_ID[c]["text"] for c in cited if c in BY_ID) + " " + ticket["question"])
    prohibited = [w for w in ticket["prohibited_claims"] if w.lower() in d]
    words = len(draft.split())
    unseen = [c for c in cited if retrieved_ids is not None and c not in retrieved_ids]
    res = dict(escalated=esc, cited_ids=cited, words=words, cited_unseen=unseen, copy_share=copy_share(draft, cited_text), citation_valid=valid,
               has_gold_citation=any(c in ticket["acceptable_source_ids"] for c in cited),
               citation_support=bool(cited) and all(supported), fact_coverage=(sum(stated) / len(groups)) if groups else 1.0,
               facts_ok=all(stated), unsupported_nums=sorted(nums), prohibited_hit=prohibited)
    if ticket["must_escalate"]:
        res["correct"] = esc
        res["failure"] = None if esc else "missed_escalation"
    else:
        reasons = []
        if esc:
            reasons.append("over_escalated")
        else:
            if not res["facts_ok"]:
                reasons.append("missing_fact")
            if not valid or not res["has_gold_citation"]:
                reasons.append("bad_citation")
            elif not res["citation_support"]:
                reasons.append("citation_not_supporting")
            if unseen:
                reasons.append("cited_unseen")
            if words > MAX_WORDS:
                reasons.append("too_long")
            if nums:
                reasons.append("unsupported_number")
            if prohibited:
                reasons.append("prohibited_claim")
        res["correct"] = not reasons
        res["failure"] = reasons[0] if reasons else None
        res["failures"] = reasons
    return res


def summarize_quality(scored, tickets_by_id):
    """scored: list of (ticket_id, score dict). Returns aggregate rates."""
    n = len(scored)
    ans = [(t, s) for t, s in scored if not tickets_by_id[t]["must_escalate"]]
    esc = [(t, s) for t, s in scored if tickets_by_id[t]["must_escalate"]]
    rate = lambda xs, f: (sum(f(s) for _, s in xs) / len(xs)) if xs else float("nan")
    return dict(
        n=n, n_answerable=len(ans), n_must_escalate=len(esc),
        correct_rate=rate(scored, lambda s: s["correct"]),
        supported_answer_rate=rate(ans, lambda s: s["correct"]),
        fact_coverage=rate(ans, lambda s: s["fact_coverage"]),
        citation_valid_rate=rate([(t, s) for t, s in ans if not s["escalated"]], lambda s: s["citation_valid"] and bool(s["cited_ids"])),
        citation_support_rate=rate([(t, s) for t, s in ans if not s["escalated"]], lambda s: s["citation_support"]),
        unsupported_number_rate=rate([(t, s) for t, s in ans if not s["escalated"]], lambda s: bool(s["unsupported_nums"])),
        copy_share=rate([(t, s) for t, s in ans if not s["escalated"]], lambda s: s.get("copy_share", 0.0)),
        prohibited_claim_rate=rate([(t, s) for t, s in ans if not s["escalated"]], lambda s: bool(s["prohibited_hit"])),
        over_escalation_rate=rate(ans, lambda s: s["escalated"]),
        escalation_recall=rate(esc, lambda s: s["escalated"]))
