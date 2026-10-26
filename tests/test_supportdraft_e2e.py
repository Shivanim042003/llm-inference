"""End-to-end: runner -> backend -> scorer -> csv -> report, against (a) a scripted OpenAI-compatible fake and (b) the real tiny-model server."""
import asyncio
import json
import re
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from llmserve.models import TINY_ID
from llmserve.server import build_app
from supportdraft import compare, report, runner
from supportdraft.kb import BY_ID
from supportdraft.tickets import TICKETS, split


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


def scripted_app(policy, delay=0.01):
    """A fake OpenAI-style streaming backend whose reply is policy(question, passage_ids)."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        user = body["messages"][-1]["content"]
        ids = re.findall(r"^\[(P\d{2})\]", user, flags=re.M)
        question = user.split("Customer ticket:\n")[1].split("\n\nDraft reply:")[0]
        text = policy(question, ids)

        async def gen():
            yield "data: " + json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}) + "\n\n"   # empty role chunk, like vLLM
            await asyncio.sleep(delay)
            for i in range(0, len(text), 12):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + 12]}}]}) + "\n\n"
                await asyncio.sleep(delay / 4)
            yield "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": len(text) // 4}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return app


def good_policy(question, ids):
    """Quote the top passage, but escalate when the question matches one of the known unsupported requests."""
    t = next(x for x in TICKETS if x["question"] == question)
    if t["must_escalate"] or not ids:
        return "ESCALATE: not covered by the approved knowledge"
    return f"Thanks for reaching out. {BY_ID[ids[0]]['text']} [{ids[0]}]"


@pytest.fixture(scope="module")
def fake():
    base, srv, th = serve(scripted_app(good_policy))
    yield base
    srv.should_exit = True
    th.join(timeout=5)


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    for mod in (compare, report):
        monkeypatch.setattr(mod, "PROC", tmp_path / "processed")
    monkeypatch.setattr(compare, "OUT", tmp_path / "sd")
    monkeypatch.setattr(report, "RAW", tmp_path / "sd")
    monkeypatch.setattr(report, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(report, "FIGS", tmp_path / "reports" / "figures")
    return tmp_path


def test_runner_scores_a_scripted_backend(fake):
    url = f"{fake}/v1/chat/completions"
    tickets = split("test")[:20] + [t for t in split("test") if t["must_escalate"]][:5]
    recs, serving, quality = runner.evaluate(url, "m", tickets, rate=20, e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    assert serving["failed"] == 0 and serving["n_requests"] == len(tickets)
    assert all(r["ttft_s"] is not None and r["e2e_s"] >= r["ttft_s"] for r in recs)      # TTFT = first NON-EMPTY chunk
    assert quality["escalation_recall"] == 1.0 and quality["over_escalation_rate"] == 0.0
    assert quality["citation_support_rate"] >= 0.9       # <1: when retrieval's top-1 is the wrong passage, citing it does not support the claim
    assert serving["useful"] > 0 and serving["on_time_rate"] == 1.0


def test_late_drafts_are_not_useful(fake):
    url = f"{fake}/v1/chat/completions"
    tickets = split("dev")[:6]
    _, serving, _ = runner.evaluate(url, "m", tickets, rate=50, e2e_slo_s=0.001, ttft_slo_s=0.001, min_seconds=0.0)
    assert serving["useful"] == 0 and serving["on_time_rate"] == 0.0


def test_backend_error_is_counted_not_hidden():
    app = FastAPI()

    @app.get("/health")
    def h():
        return {}

    @app.post("/v1/chat/completions")
    async def chat():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "boom"}, status_code=503)
    base, srv, th = serve(app)
    try:
        recs, serving, quality = runner.evaluate(f"{base}/v1/chat/completions", "m", split("dev")[:5], rate=50, e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
        assert serving["failed"] == 5 and serving["useful"] == 0 and quality["requests_failed"] == 5
        assert {r["error"] for r in recs} == {"http 503"}
    finally:
        srv.should_exit = True
        th.join(timeout=5)


def test_full_pipeline_writes_csv_jsonl_and_report(fake, tmp_dirs):
    rows = compare.run_on_url("scripted", f"{fake}/v1/chat/completions", "m", rates=[10, 40], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    assert len(rows) == 2
    (tmp_dirs / "processed" / "supportdraft_setup_test.json").write_text(json.dumps(dict(model="m", e2e_slo_s=5, ttft_slo_s=1)))
    assert (tmp_dirs / "sd" / "scripted_test_rate10.jsonl").exists()
    md = report.build()
    assert "SYNTHETIC" in md.upper() and "baseline: quote top-1 passage" in md and "scripted" in md
    assert "ESTIMATE" in md and "no reviewer time" in md.lower()
    assert (tmp_dirs / "reports" / "SUPPORTDRAFT.md").exists() and (tmp_dirs / "reports" / "figures" / "supportdraft.png").exists()
    # re-running replaces that backend's rows rather than duplicating them
    compare.run_on_url("scripted", f"{fake}/v1/chat/completions", "m", rates=[10], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    import pandas as pd
    assert len(pd.read_csv(tmp_dirs / "processed" / "supportdraft_serving.csv")) == 1


def test_review_sheet_roundtrip(fake, tmp_dirs):
    from supportdraft.review import make_review_csv, summarize_review
    import pandas as pd
    compare.run_on_url("scripted", f"{fake}/v1/chat/completions", "m", rates=[20], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    path = make_review_csv(tmp_dirs / "sd" / "scripted_test_rate20.jsonl", n=10)
    df = pd.read_csv(path, dtype=str).fillna("")
    assert len(df) == 10 and set(df.columns) >= {"ticket_id", "question", "draft", "automated_correct", "human_verdict"}
    df["human_verdict"] = ["approve"] * 6 + ["edit"] * 2 + ["reject"] * 2
    df["review_seconds"] = "12"
    df.to_csv(path, index=False)
    s = summarize_review(path)
    assert s["n_reviewed"] == 10 and abs(s["approve_rate"] - 0.6) < 1e-9 and s["mean_review_seconds"] == 12
    assert 0 <= s["automated_vs_human_agreement"] <= 1
    assert summarize_review(path)["n_reviewed"] == 10


def test_real_llmserve_server_speaks_the_protocol_the_runner_expects():
    """The tiny random model writes nonsense, so quality is ~0; this checks streaming, usage, EOS handling and TTFT with the real server."""
    app = build_app(TINY_ID, mode="continuous", max_batch=8, dtype="float32", engine_kind="slot", max_ctx=8192)
    base, srv, th = serve(app)
    try:
        recs, serving, quality = runner.evaluate(f"{base}/v1/chat/completions", TINY_ID, split("dev")[:6], rate=5, e2e_slo_s=60, ttft_slo_s=30,
                                                 min_seconds=0.0, max_tokens=12)
        assert serving["failed"] == 0, [r["error"] for r in recs]
        assert all(r["output_tokens"] and r["output_tokens"] <= 12 for r in recs)
        assert all(r["ttft_s"] is not None for r in recs)
        assert quality["supported_answer_rate"] == 0.0         # nonsense text must not score as correct
    finally:
        srv.should_exit = True
        th.join(timeout=5)


def test_readme_summary_is_generated_from_data(fake, tmp_dirs):
    compare.run_on_url("scripted", f"{fake}/v1/chat/completions", "m", rates=[10], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    (tmp_dirs / "processed" / "supportdraft_setup_test.json").write_text(json.dumps(dict(model="m", e2e_slo_s=5, ttft_slo_s=1)))
    report.SUMMARY.clear()
    report.build()
    md = report.summary_markdown()
    assert "scripted" in md and "synthetic" in md.lower() and "assumed GPU price" in md and "baseline: always escalate" in md


def test_report_ignores_dev_rows(fake, tmp_dirs):
    url = f"{fake}/v1/chat/completions"
    compare.run_on_url("scripted", url, "m", rates=[10], split_name="dev", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    with pytest.raises(SystemExit):
        report.build()                                # only dev rows exist -> refuse to report them as test results
    compare.run_on_url("scripted", url, "m", rates=[10], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    md = report.build()
    assert "split `test`" in md
    import pandas as pd
    assert set(pd.read_csv(tmp_dirs / "processed" / "supportdraft_serving.csv")["split"]) == {"dev", "test"}


def test_interactive_review_records_verdicts_and_resumes(fake, tmp_dirs, monkeypatch):
    import pandas as pd
    from supportdraft.review import make_review_csv, run_interactive, summarize_review
    compare.run_on_url("scripted", f"{fake}/v1/chat/completions", "m", rates=[20], split_name="test", e2e_slo_s=5, ttft_slo_s=1, min_seconds=0.0)
    path = make_review_csv(tmp_dirs / "sd" / "scripted_test_rate20.jsonl", n=6)
    answers = iter(["x", "a", "e", "r", "s", "q"])               # 'x' is invalid and must be re-asked; 's' skips; 'q' quits
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    run_interactive(path)
    df = pd.read_csv(path, dtype=str).fillna("")
    assert list(df["human_verdict"][:3]) == ["approve", "edit", "reject"] and df["human_verdict"][3] == ""
    assert summarize_review(path)["n_reviewed"] == 3
    answers = iter(["a", "a", "a"])                              # resume: already reviewed rows are not asked again
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    run_interactive(path)
    assert summarize_review(path)["n_reviewed"] == 6
