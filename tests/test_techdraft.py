import asyncio
import json
import re
import socket
import threading
import time

import httpx
import pandas as pd
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from llmserve.models import TINY_ID
from llmserve.server import build_app
from techdraft import compare, data, pipeline, report, run, score

TOPICS = ["printer", "router", "database", "firewall", "scheduler", "backup", "gateway", "queue", "cache", "certificate"]


def make_records(n_ans=40, n_imp=20):
    recs = []
    for i in range(n_ans):
        t = TOPICS[i % len(TOPICS)]
        key = f"zq{i}w"
        filler = " ".join(f"background{j} about general maintenance of the {t} subsystem" for j in range(60))
        fact = f"To fix the {t} fault code {key} restart the {t} service and clear the {t} lock file under the var directory."
        text = f"{filler} {fact} {filler}"
        q = f"My {t} shows fault code {key}. How do I fix it?"
        recs.append(dict(id=f"A{i:03d}", question=q, answer=fact, is_impossible=False, contexts=[dict(filename=f"doc{i:03d}.txt", text=text)]))
    for i in range(n_imp):
        recs.append(dict(id=f"I{i:03d}", question=f"Can you tell me about the weather and lunch menu number {i}?", answer="", is_impossible=True, contexts=[]))
    return recs


@pytest.fixture
def recs():
    return make_records()


def write_jsonl(recs, path):
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- data
def test_chunks_are_exact_slices_with_offsets_and_overlap():
    text = "alpha  beta\n gamma " * 200
    ch = data.chunk_doc("d.txt", text, words=50, overlap=10)
    assert len(ch) > 3
    for c in ch:
        assert text[c["start"]:c["end"]] == c["text"]
    assert ch[1]["start"] < ch[0]["end"]                       # overlap
    assert ch[-1]["end"] == len(text.rstrip())                 # covers the whole document


def test_split_is_deterministic_and_roughly_40_percent():
    ids = [f"Q{i}" for i in range(2000)]
    s = [data.split_of(i) for i in ids]
    assert s == [data.split_of(i) for i in ids]
    assert 0.35 < s.count("tune") / len(s) < 0.45


def test_load_normalises_fields(tmp_path):
    p = write_jsonl([dict(id=1, question="q", answer=None, is_impossible=1, contexts=None)], tmp_path / "x.jsonl")
    r = data.load(p)[0]
    assert r["id"] == "1" and r["answer"] == "" and r["is_impossible"] is True and r["contexts"] == []


def test_retrieval_finds_the_gold_document(recs):
    kb = data.KB(recs)
    ans = [r for r in recs if not r["is_impossible"]]
    hits = sum(any(kb.by_id[c]["doc"] in data.gold_docs(r) for c, _ in kb.retrieve(r["question"], 4)) for r in ans)
    assert hits / len(ans) >= 0.95
    assert kb.retrieve("zzzzzz qqqqqq") == []


# ---------------------------------------------------------------- pipeline
def chunks():
    return [dict(id="C00001", doc="a", text="Restart the service and clear the lock file."), dict(id="C00002", doc="b", text="Unrelated text about weather.")]


def test_parse_and_verify_grounded_quote():
    out = "DECISION: ANSWER\nEVIDENCE: \"Restart the service and clear the lock file.\" [C00001]\nANSWER: Restart the service and clear the lock file."
    p = pipeline.parse(out)
    assert p["decision"] == "ANSWER" and p["quotes"] == [("Restart the service and clear the lock file.", "C00001")] and p["answer"].startswith("Restart")
    assert pipeline.verify(p, chunks()) == (1, 1)


def test_verify_rejects_invented_wrong_chunk_and_unseen():
    inv = pipeline.parse("DECISION: ANSWER\nEVIDENCE: Reboot the whole server. [C00001]\nANSWER: x")
    assert pipeline.verify(inv, chunks()) == (1, 0)
    wrong = pipeline.parse("DECISION: ANSWER\nEVIDENCE: Restart the service and clear the lock file. [C00002]\nANSWER: x")
    assert pipeline.verify(wrong, chunks()) == (1, 0)                      # real sentence, but not in the excerpt it cites
    unseen = pipeline.parse("DECISION: ANSWER\nEVIDENCE: Restart the service and clear the lock file. [C00099]\nANSWER: x")
    assert pipeline.verify(unseen, chunks()) == (1, 0)


def test_parse_abstain_and_unparsed():
    a = pipeline.parse("DECISION: INSUFFICIENT_EVIDENCE\nEVIDENCE: none\nANSWER: none")
    assert a["decision"] == "INSUFFICIENT" and a["quotes"] == [] and a["answer"] == ""
    assert pipeline.parse("Sure! Just restart it.")["decision"] == "UNPARSED"
    assert pipeline.parse("decision: insufficient evidence")["decision"] == "INSUFFICIENT"
    assert pipeline.parse("INSUFFICIENT_EVIDENCE")["decision"] == "INSUFFICIENT"                     # label dropped
    assert pipeline.parse("EVIDENCE: none\n\nANSWER: None")["decision"] == "INSUFFICIENT"
    assert pipeline.parse("The INSUFFICIENT_EVIDENCE of ...")["decision"] == "UNPARSED"
    assert pipeline.parse("EVIDENCE: none ANSWER: INSUFFICIENT_EVIDENCE")["decision"] == "INSUFFICIENT"
    assert pipeline.parse("ANSWER: INSUFFICIENT_EVIDENCE")["decision"] == "INSUFFICIENT"
    assert pipeline.parse("EVIDENCE: none\nANSWER: Restart it.")["decision"] == "ANSWER"               # label dropped, but it answered
    b = pipeline.parse("DECISION: ANSWER\nEVIDENCE: [C00001] \"Restart the service and clear the lock file.\"\nANSWER: x")
    assert b["quotes"] == [("Restart the service and clear the lock file.", "C00001")]                # id before quote
    assert pipeline.verify(b, chunks()) == (1, 1)
    n = pipeline.parse("DECISION: ANSWER\nEVIDENCE: \"Restart the service and clear the lock file.\"\nEVIDENCE: \"Reboot the whole server now please.\"\nANSWER: x")
    assert [c for _, c in n["quotes"]] == [None, None]
    assert pipeline.verify(n, chunks()) == (2, 1)                                                      # no id: real in any shown excerpt, invented is not


def test_prompt_contains_format_excerpt_ids_and_question():
    m = pipeline.build_messages("How?", chunks())
    assert "INSUFFICIENT_EVIDENCE" in m[0]["content"] and "ONLY" in m[0]["content"]
    assert "[C00001]" in m[1]["content"] and "How?" in m[1]["content"]
    assert "(no excerpts found)" in pipeline.build_messages("x", [])[1]["content"]


# ------------------------------------------------------------------ score
def item(imp=False, gold="Restart the service and clear the lock file."):
    return dict(impossible=imp, gold_answer=gold, gold_docs={"a"}, retrieved=[("C00001", 5.0)])


def test_f1_and_pass_rules():
    good = dict(decision="ANSWER", quotes=[("Restart the service and clear the lock file.", "C00001")], answer="Restart the service and clear the lock file.")
    s = score.score_item(item(), good, chunks())
    assert s["passed"] and s["f1"] > 0.9 and s["grounded"] and s["retrieval_hit"]
    ungrounded = dict(good, quotes=[("Reboot the server now.", "C00001")])
    assert not score.score_item(item(), ungrounded, chunks())["passed"]
    off_topic = dict(good, answer="Please contact your administrator about billing.")
    assert not score.score_item(item(), off_topic, chunks())["passed"]
    assert not score.score_item(item(), dict(decision="INSUFFICIENT", quotes=[], answer=""), chunks())["passed"]     # over-abstained
    assert score.score_item(item(imp=True, gold=""), dict(decision="INSUFFICIENT", quotes=[], answer=""), chunks())["passed"]
    assert not score.score_item(item(imp=True, gold=""), good, chunks())["passed"]                                    # answered an unanswerable one


def test_blind_baselines_do_not_score_well_everywhere(recs):
    kb = data.KB(recs)
    items = run.prepare_items(recs, kb)
    b = score.score_baselines(items, kb, threshold=0.0)
    ab = b["baseline: always abstain"]
    assert ab["answerable_pass"] == 0 and ab["impossible_pass"] == 1
    top = b["baseline: quote top-1 chunk (no LLM)"]
    assert top["impossible_pass"] == 0                                # never abstains


def test_threshold_tuning_separates_answerable_from_impossible(recs):
    kb = data.KB(recs)
    items = run.prepare_items(recs, kb)
    th = score.tune_threshold(items, kb)
    m = score.score_baselines(items, kb, th)["baseline: top-1 chunk + score threshold (no LLM)"]
    assert m["impossible_pass"] >= 0.9 and m["answerable_pass"] >= 0.5


# ------------------------------------------------------------- end to end
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app):
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    return f"http://127.0.0.1:{port}", srv, th


def scripted_app(policy):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        user = body["messages"][-1]["content"]
        question = user.split("Question:\n")[1].split("\n\nReply:")[0]
        excerpts = re.findall(r"^\[(C\d{5})\] (.+?)(?=\n\n\[C|\n\nQuestion:)", user, flags=re.M | re.S)
        text = policy(question, excerpts)

        async def gen():
            yield "data: " + json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}) + "\n\n"
            for i in range(0, len(text), 16):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + 16]}}]}) + "\n\n"
                await asyncio.sleep(0.002)
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": len(text) // 4}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return app


def good_policy(question, excerpts):
    if "weather" in question or not excerpts:
        return "DECISION: INSUFFICIENT_EVIDENCE\nEVIDENCE: none\nANSWER: none"
    cid, text = excerpts[0]
    m = re.search(r"To fix the.*?under the var directory\.", text, flags=re.S)
    sent = m.group(0) if m else text[:80]
    return f"DECISION: ANSWER\nEVIDENCE: {sent} [{cid}]\nANSWER: {sent}"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "PROC", tmp_path / "processed")
    monkeypatch.setattr(compare, "RAW", tmp_path / "raw")
    monkeypatch.setattr(report, "PROC", tmp_path / "processed")
    monkeypatch.setattr(report, "REPORTS", tmp_path / "reports")
    return tmp_path


def test_scripted_backend_end_to_end_and_report(paths, recs):
    jp = write_jsonl(recs, paths / "d.jsonl")
    base, srv, th = serve(scripted_app(good_policy))
    try:
        kb = data.KB(recs)
        items = run.prepare_items([r for r in recs if data.split_of(r["id"]) == "final"], kb)
        url = f"{base}/v1/chat/completions"
        compare.run_on_url("scripted", url, "m", [20], items, kb, "final", "quality", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0)
        compare.run_on_url("scripted", url, "m", [10, 30], items, kb, "final", "sweep", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0, max_items=15)
        tune_items = run.prepare_items([r for r in recs if data.split_of(r["id"]) == "tune"], kb)
        compare.run_on_url("scripted", url, "m", [20], tune_items, kb, "tune", "quality", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0)
    finally:
        srv.should_exit = True
        th.join(timeout=5)
    q = pd.read_csv(paths / "processed" / "techdraft_quality.csv")
    row = q[(q.split == "final") & (q.run == "quality")].iloc[0]
    assert row["pass_rate"] > 0.9 and row["impossible_pass"] == 1.0 and row["grounded_rate"] == 1.0 and row["requests_failed"] == 0
    s = pd.read_csv(paths / "processed" / "techdraft_serving.csv")
    assert len(s[(s.run == "sweep")]) == 2 and (s[(s.run == "sweep")].n_unique == 15).all()
    assert list((paths / "raw").glob("scripted_final_quality_rate20.jsonl"))
    md = report.build(data_path=jp)
    assert "baseline: always abstain" in md and "scripted" in md and "Sustainable rate" in md and "not** in this repository" in md
    assert (paths / "reports" / "TECHDRAFT.md").exists()


def test_rerun_replaces_only_the_same_run(paths, recs):
    kb = data.KB(recs)
    items = run.prepare_items([r for r in recs if data.split_of(r["id"]) == "final"][:6], kb)
    base, srv, th = serve(scripted_app(good_policy))
    try:
        url = f"{base}/v1/chat/completions"
        compare.run_on_url("s", url, "m", [20], items, kb, "final", "quality", min_seconds=0)
        compare.run_on_url("s", url, "m", [10], items, kb, "final", "sweep", min_seconds=0)
        compare.run_on_url("s", url, "m", [20], items, kb, "final", "quality", min_seconds=0)
    finally:
        srv.should_exit = True
        th.join(timeout=5)
    s = pd.read_csv(paths / "processed" / "techdraft_serving.csv")
    assert sorted(s.run) == ["quality", "sweep"]


def test_backend_errors_are_counted(paths, recs):
    app = FastAPI()

    @app.get("/health")
    def h():
        return {}

    @app.post("/v1/chat/completions")
    async def chat():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "boom"}, status_code=503)
    base, srv, th = serve(app)
    kb = data.KB(recs)
    items = run.prepare_items(recs[:5], kb)
    try:
        r, serving, quality = run.evaluate(f"{base}/v1/chat/completions", "m", items, kb, 50, 5, 1, min_seconds=0)
        assert serving["failed"] == 5 and serving["useful"] == 0 and quality["requests_failed"] == 5
        assert {x["error"] for x in r} == {"http 503"}
    finally:
        srv.should_exit = True
        th.join(timeout=5)


def test_real_llmserve_server_speaks_the_protocol(paths, recs):
    app = build_app(TINY_ID, mode="continuous", max_batch=8, dtype="float32", engine_kind="slot", max_ctx=8192)
    base, srv, th = serve(app)
    kb = data.KB(recs)
    items = run.prepare_items(recs[:6], kb)
    try:
        r, serving, quality = run.evaluate(f"{base}/v1/chat/completions", TINY_ID, items, kb, 5, 60, 30, min_seconds=0, max_tokens=12)
        assert serving["failed"] == 0, [x["error"] for x in r]
        assert all(x["ttft_s"] is not None for x in r)
        assert quality["answerable_pass"] == 0.0 or quality["answerable_pass"] != quality["answerable_pass"]    # nonsense text must not pass
    finally:
        srv.should_exit = True
        th.join(timeout=5)
