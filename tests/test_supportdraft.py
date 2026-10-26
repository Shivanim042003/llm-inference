import math

import pytest

from supportdraft.cost import is_useful, saturated_cost, scenario, serving_summary, sustainable_rate
from supportdraft.kb import BY_ID, KB
from supportdraft.retrieval import BM25, build_messages, retrieval_recall, retrieve
from supportdraft.score import is_escalation, score_draft, summarize_quality
from supportdraft.tickets import TICKETS, split

T = {t["ticket_id"]: t for t in TICKETS}


def ticket(pid, q_contains):
    return next(t for t in TICKETS if t["acceptable_source_ids"] == [pid] and q_contains in t["question"])


# ------------------------------------------------------------------ dataset validity
def test_dataset_shape_and_ids():
    assert len(KB) == 30 and len({p["id"] for p in KB}) == 30
    assert len(TICKETS) == 114 and len({t["ticket_id"] for t in TICKETS}) == 114
    assert {t["split"] for t in TICKETS} == {"dev", "test"}
    for s in ("dev", "test"):
        kinds = {t["kind"] for t in split(s)}
        assert "escalate" in kinds and len(kinds) >= 3
    assert {t["kind"] for t in split("test")} >= {"hard", "escalate"}      # held-out set includes the hard cases
    assert not any(t["kind"] == "hard" for t in split("dev"))


def test_every_required_fact_is_supported_by_its_source():
    for t in TICKETS:
        text = " ".join(BY_ID[s]["text"] for s in t["acceptable_source_ids"]).lower()
        for g in t["required_facts"]:
            assert any(a in text for a in g), (t["ticket_id"], g)


def test_escalation_tickets_have_no_sources_and_answerable_ones_do():
    for t in TICKETS:
        assert t["must_escalate"] == (t["kind"] == "escalate")
        assert bool(t["acceptable_source_ids"]) != t["must_escalate"]
        assert t["required_facts"] or t["must_escalate"] or t["acceptable_source_ids"]


def test_prohibited_claims_never_in_the_source_passage_or_the_question():
    for t in TICKETS:
        src = " ".join(BY_ID[s]["text"] for s in t["acceptable_source_ids"]).lower()
        for w in t["prohibited_claims"]:
            assert w.lower() not in src and w.lower() not in t["question"].lower(), (t["ticket_id"], w)


def test_a_reference_answer_built_from_the_passage_scores_correct_for_every_ticket():
    """Sanity: the checks are satisfiable. Quote the facts from the source and cite it."""
    for t in TICKETS:
        if t["must_escalate"]:
            continue
        pid = t["acceptable_source_ids"][0]
        draft = f"Hello! {BY_ID[pid]['text']} [{pid}]"
        s = score_draft(t, draft)
        assert s["correct"], (t["ticket_id"], s)


def test_dataset_cannot_be_passed_by_blind_strategies():
    """Always-escalate, and dumping the whole knowledge base with every citation, must not score well."""
    ans = [t for t in TICKETS if not t["must_escalate"]]
    esc = [t for t in TICKETS if t["must_escalate"]]
    assert sum(score_draft(t, "ESCALATE: needs a person")["correct"] for t in ans) == 0
    dump = " ".join(f"{p['text']} [{p['id']}]" for p in KB)
    assert sum(score_draft(t, dump, retrieve(t["question"], 3))["correct"] for t in ans) == 0     # too long / cites unseen passages
    assert sum(score_draft(t, dump, retrieve(t["question"], 3))["correct"] for t in esc) == 0


def test_citing_a_passage_the_model_never_saw_is_flagged():
    t = ticket("P01", "how long does the reset link stay valid")
    d = "The reset link is valid for 30 minutes [P01]."
    assert score_draft(t, d, ["P01", "P05"])["correct"]
    r = score_draft(t, d, ["P05", "P06"])            # P01 was not in the context
    assert r["failure"] == "cited_unseen"


# ------------------------------------------------------------------ scorer behaviour
def test_valid_looking_citation_is_not_enough():
    t = ticket("P01", "how long does the reset link stay valid")
    good = "The reset link is valid for 30 minutes [P01]."
    wrong_cite = "The reset link is valid for 30 minutes [P05]."       # real id, but that passage does not say it
    no_cite = "The reset link is valid for 30 minutes."
    fake_cite = "The reset link is valid for 30 minutes [P99]."
    assert score_draft(t, good)["correct"]
    r = score_draft(t, wrong_cite)
    assert not r["correct"] and r["failure"] == "bad_citation"        # P05 is not an acceptable source
    assert score_draft(t, no_cite)["failure"] == "bad_citation"
    assert score_draft(t, fake_cite)["failure"] == "bad_citation" and not score_draft(t, fake_cite)["citation_valid"]


def test_gold_citation_plus_a_second_passage_that_does_not_support_the_fact():
    t = ticket("P02", "When can I sign in again")
    d = "You can sign in again after 15 minutes [P02]. Also, invitations expire after 7 days [P06]."
    assert score_draft(t, d)["correct"]            # extra cited passage supports its own number -> fine
    d2 = "You can sign in again after 15 minutes [P02]. Also, links expire after 99 days [P06]."
    r = score_draft(t, d2)
    assert not r["correct"] and r["failure"] == "unsupported_number"


def test_invented_numbers_and_prohibited_values_are_caught():
    t = ticket("P08", "Pro plan cost")
    assert score_draft(t, "Pro costs $8 per user per month [P08].")["correct"]
    r = score_draft(t, "Pro costs $8 per user per month, plus a $25 setup fee [P08].")
    assert r["failure"] == "unsupported_number" and "$25" in r["unsupported_nums"]
    r = score_draft(t, "Pro costs $8 per user per month [P08] ($10 per user for new accounts).")
    assert "prohibited_claim" in r["failures"] or "unsupported_number" in r["failures"]


def test_missing_fact_over_escalation_and_missed_escalation():
    t = ticket("P01", "how long does the reset link stay valid")
    assert score_draft(t, "Use the Forgot password link [P01].")["failure"] == "missing_fact"
    assert score_draft(t, "ESCALATE: not sure")["failure"] == "over_escalated"
    e = next(x for x in TICKETS if x["must_escalate"])
    assert score_draft(e, "ESCALATE: account security incident")["correct"]
    assert score_draft(e, "Please reset your password [P01].")["failure"] == "missed_escalation"


def test_escalation_detection_is_robust_to_formatting():
    for s in ("ESCALATE: x", "  escalate - x", "**ESCALATE**: x", "`ESCALATE: x`", "\"ESCALATE: x\""):
        assert is_escalation(s), s
    assert not is_escalation("Please escalate this yourself if needed.")


def test_list_markers_and_passage_ids_are_not_counted_as_numbers():
    t = ticket("P03", "Where do I turn on two-factor")
    d = "1. Open Settings > Security [P03].\n2. Follow the steps."
    assert score_draft(t, d)["correct"], score_draft(t, d)


def test_conflicting_premise_is_not_echoed_credit():
    t = next(x for x in TICKETS if x["kind"] == "hard" and x["acceptable_source_ids"] == ["P10"])
    ok = "Monthly plans are not refundable, but you can cancel any time [P10]. Refund requests go to the Billing team."
    bad = "Sure, you have 60 days to ask for a refund [P10]."
    assert score_draft(t, ok)["correct"]
    assert not score_draft(t, bad)["correct"]


def test_summary_rates():
    sc = [(t["ticket_id"], score_draft(t, "ESCALATE: x")) for t in TICKETS[:3] + [x for x in TICKETS if x["must_escalate"]][:2]]
    q = summarize_quality(sc, T)
    assert q["over_escalation_rate"] == 1.0 and q["escalation_recall"] == 1.0 and q["supported_answer_rate"] == 0.0


# ------------------------------------------------------------------ retrieval
def test_bm25_basics():
    assert retrieve("how do I reset my password")[0] == "P01"
    assert retrieve("zzzz qqqq xxxx") == []
    assert "P22" in retrieve("csv import rows limit", 3)
    assert BM25(KB).top("api rate limit 429", 1) == ["P21"]


def test_retrieval_recall_floor_on_dev_and_test():
    assert retrieval_recall(split("dev"), 3) >= 0.9
    assert retrieval_recall(split("test"), 3) >= 0.9


def test_prompt_contains_rules_passages_and_ticket():
    msgs = build_messages("How long is the link valid?", ["P01", "P05"])
    assert msgs[0]["role"] == "system" and "ESCALATE" in msgs[0]["content"] and "ONLY" in msgs[0]["content"]
    u = msgs[1]["content"]
    assert "[P01]" in u and "[P05]" in u and "How long is the link valid?" in u
    assert "(no passages found)" in build_messages("x", [])[1]["content"]


# ------------------------------------------------------------------ cost and capacity arithmetic
def test_useful_requires_ok_correct_and_on_time():
    base = dict(ok=True, correct=True, e2e_s=3.0)
    assert is_useful(base, 5.0)
    assert not is_useful({**base, "e2e_s": 6.0}, 5.0)           # a fast first token does not make a late draft useful
    assert not is_useful({**base, "correct": False}, 5.0)
    assert not is_useful({**base, "ok": False, "e2e_s": None}, 5.0)


def test_serving_summary_and_costs():
    recs = [dict(ok=True, correct=True, e2e_s=2.0, ttft_s=0.5, output_tokens=100),
            dict(ok=True, correct=False, e2e_s=3.0, ttft_s=0.6, output_tokens=100),
            dict(ok=True, correct=True, e2e_s=20.0, ttft_s=0.7, output_tokens=100),
            dict(ok=False, correct=False, e2e_s=None, ttft_s=None, output_tokens=None)]
    s = serving_summary(recs, wall_s=10.0, e2e_slo_s=5.0, ttft_slo_s=1.0)
    assert s["failed"] == 1 and s["useful"] == 1 and s["on_time_rate"] == 0.5 and s["ttft_ok_rate"] == 0.75
    assert math.isclose(s["output_tok_per_s"], 30.0)
    c = saturated_cost(useful=1000, wall_s=3600, gpu_usd_per_hour=0.5)
    assert math.isclose(c["usd_per_1k_useful"], 0.5)
    assert saturated_cost(0, 10, 1.0)["usd_per_1k_useful"] == float("inf")


def test_sustainable_rate_needs_every_condition():
    rows = [dict(offered_rps=1, failed=0, on_time_rate=1.0), dict(offered_rps=2, failed=0, on_time_rate=0.97),
            dict(offered_rps=4, failed=0, on_time_rate=0.80), dict(offered_rps=8, failed=3, on_time_rate=1.0)]
    assert sustainable_rate(rows) == (2, 4)
    assert sustainable_rate([dict(offered_rps=1, failed=0, on_time_rate=0.5)]) == (0.0, 1)


def test_scenario_is_arithmetic_on_stated_assumptions():
    s = scenario(sustainable_rps=2.0, useful_rate=0.8, gpu_usd_per_hour=0.4, agents=100,
                 drafts_per_agent_per_hour=6, service_hours_per_day=10)
    assert math.isclose(s["agents_supported"], 2.0 / (6 / 3600))
    assert math.isclose(s["daily_gpu_usd"], 4.0) and math.isclose(s["daily_useful"], 100 * 6 * 10 * 0.8)
    assert math.isclose(s["usd_per_1k_useful_provisioned"], 1000 * 4.0 / 4800)
    # a lightly used service is costlier per useful draft than a busy one on the same GPU
    busy = scenario(2.0, 0.8, 0.4, 1000, 6, 10)
    assert busy["usd_per_1k_useful_provisioned"] < s["usd_per_1k_useful_provisioned"]
