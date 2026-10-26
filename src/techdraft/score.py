"""Automated scoring against the gold answer, plus no-LLM baselines. These are rule-based overlap measures, not human judgments."""
from __future__ import annotations

import re
from collections import Counter

from supportdraft.retrieval import STOP

from .pipeline import verify

F1_PASS = 0.20          # an answerable item passes only if its answer overlaps the gold answer at least this much (token F1)


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP]


def f1_recall(pred, gold):
    p, g = Counter(toks(pred)), Counter(toks(gold))
    if not p or not g:
        return 0.0, 0.0
    ov = sum((p & g).values())
    if ov == 0:
        return 0.0, 0.0
    pr, rc = ov / sum(p.values()), ov / sum(g.values())
    return 2 * pr * rc / (pr + rc), rc


def score_item(item, parsed, shown_chunks):
    """item: dict(impossible, gold_answer, gold_docs, retrieved [(id, score)]). Returns per-item metrics."""
    nq, ng = verify(parsed, shown_chunks)
    grounded = nq > 0 and ng == nq
    dec = parsed["decision"]
    f1, _ = f1_recall(parsed["answer"], item["gold_answer"])
    _, ev_rc = f1_recall(" ".join(q for q, _ in parsed["quotes"]), item["gold_answer"])
    hit = any(c["doc"] in item["gold_docs"] for c in shown_chunks)
    if item["impossible"]:
        passed = dec == "INSUFFICIENT"
    else:
        passed = dec == "ANSWER" and grounded and f1 >= F1_PASS
    return dict(decision=dec, n_quotes=nq, n_grounded=ng, grounded=grounded, f1=f1, evidence_recall=ev_rc, retrieval_hit=hit,
                impossible=item["impossible"], passed=passed)


def summarize(scores):
    """scores: list of per-item dicts from score_item."""
    ans = [s for s in scores if not s["impossible"]]
    imp = [s for s in scores if s["impossible"]]
    answered = [s for s in ans if s["decision"] == "ANSWER"]
    abst = lambda s: s["decision"] != "ANSWER"            # INSUFFICIENT or unparsed: no usable answer was given
    tp = sum(abst(s) for s in imp)
    fp = sum(abst(s) for s in ans)
    m = lambda xs, f: (sum(f(x) for x in xs) / len(xs)) if xs else float("nan")
    return dict(n=len(scores), n_answerable=len(ans), n_impossible=len(imp),
                pass_rate=m(scores, lambda s: s["passed"]),
                answerable_pass=m(ans, lambda s: s["passed"]),
                impossible_pass=m(imp, lambda s: s["passed"]),
                mean_f1_all_answerable=m(ans, lambda s: s["f1"] if s["decision"] == "ANSWER" else 0.0),
                mean_f1_when_answered=m(answered, lambda s: s["f1"]),
                evidence_recall=m(answered, lambda s: s["evidence_recall"]),
                grounded_rate=m(answered, lambda s: s["grounded"]),
                retrieval_hit=m(ans, lambda s: s["retrieval_hit"]),
                abstain_precision=(tp / (tp + fp)) if (tp + fp) else float("nan"),
                abstain_recall=(tp / len(imp)) if imp else float("nan"),
                unparsed_rate=m(scores, lambda s: s["decision"] == "UNPARSED"))


# ---------------------------------------------------------------- no-LLM baselines (scored with the same function)
def baseline_always_abstain(item, shown):
    return dict(decision="INSUFFICIENT", quotes=[], answer="")


def baseline_top_chunk(item, shown):
    if not shown:
        return dict(decision="INSUFFICIENT", quotes=[], answer="")
    t = shown[0]
    return dict(decision="ANSWER", quotes=[(t["text"], t["id"])], answer=t["text"])


def make_threshold_baseline(threshold):
    def fn(item, shown):
        if not shown or not item["retrieved"] or item["retrieved"][0][1] < threshold:
            return dict(decision="INSUFFICIENT", quotes=[], answer="")
        return baseline_top_chunk(item, shown)
    return fn


def tune_threshold(items, kb, k=4):
    """Pick the top-1 BM25 score threshold that maximises balanced accuracy (answerable answered + impossible abstained) on TUNE items."""
    cand = sorted({round(it["retrieved"][0][1], 3) for it in items if it["retrieved"]})
    best = (-1.0, 0.0)
    for th in [0.0] + cand:
        fn = make_threshold_baseline(th)
        sc = []
        for it in items:
            shown = [kb.by_id[c] for c, _ in it["retrieved"][:k]]
            sc.append(score_item(it, fn(it, shown), shown))
        m = summarize(sc)
        bal = 0.5 * (m["answerable_pass"] + m["impossible_pass"])
        if bal > best[0]:
            best = (bal, th)
    return best[1]


def score_baselines(items, kb, threshold, k=4):
    out = {}
    for name, fn in (("baseline: always abstain", baseline_always_abstain), ("baseline: quote top-1 chunk (no LLM)", baseline_top_chunk),
                     ("baseline: top-1 chunk + score threshold (no LLM)", make_threshold_baseline(threshold))):
        sc = []
        for it in items:
            shown = [kb.by_id[c] for c, _ in it["retrieved"][:k]]
            sc.append(score_item(it, fn(it, shown), shown))
        out[name] = summarize(sc)
    return out
