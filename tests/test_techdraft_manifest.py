import json

import pandas as pd

from techdraft import __main__ as cli
from techdraft import compare, data, manifest, report, run
from test_techdraft import make_records, serve, scripted_app, good_policy, write_jsonl


def test_manifest_hashes_are_stable_and_record_exclusions(tmp_path):
    recs = make_records()
    kb = data.KB(recs)
    items = run.prepare_items(recs, kb, max_prompt_tokens=None)
    tune = [i for i in items if i["split"] == "tune"]
    fin = [i for i in items if i["split"] == "final"]
    m1 = manifest.build(recs, {"tune": tune, "final": fin}, {"tune": ["X1"], "final": []})
    m2 = manifest.build(recs, {"tune": list(reversed(tune)), "final": fin}, {"tune": ["X1"], "final": []})
    assert m1["splits"]["tune"]["ids_sha256_16"] == m2["splits"]["tune"]["ids_sha256_16"]          # order does not matter
    assert m1["splits"]["tune"]["excluded_ids"] == ["X1"] and m1["splits"]["final"]["excluded_ids"] == []
    assert m1["splits"]["final"]["n"] == len(fin) and len(m1["system_prompt_sha256_16"]) == 16
    m3 = manifest.build(recs, {"tune": tune[:-1], "final": fin}, {})
    assert m3["splits"]["tune"]["ids_sha256_16"] != m1["splits"]["tune"]["ids_sha256_16"]          # a different id set changes the hash
    p = manifest.write(m1, tmp_path / "m.json")
    assert json.loads(p.read_text())["dataset"] == "nvidia/TechQA-RAG-Eval"


def test_report_writes_manifest_and_lists_exclusions(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "PROC", tmp_path / "processed"); monkeypatch.setattr(compare, "RAW", tmp_path / "raw")
    monkeypatch.setattr(report, "PROC", tmp_path / "processed"); monkeypatch.setattr(report, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(manifest, "PROC", tmp_path / "processed")
    recs = make_records()
    jp = write_jsonl(recs, tmp_path / "d.jsonl")
    kb = data.KB(recs)
    items = run.prepare_items([r for r in recs if data.split_of(r["id"]) == "final"], kb, max_prompt_tokens=None)
    base, srv, th = serve(scripted_app(good_policy))
    try:
        compare.run_on_url("scripted", f"{base}/v1/chat/completions", "m", [20], items, kb, "final", "quality", min_seconds=0)
    finally:
        srv.should_exit = True
        th.join(timeout=5)
    md = report.build(data_path=jp)
    assert "Excluded ids" in md and "techdraft_manifest.json" in md and "does not check that every statement" in md
    assert (tmp_path / "processed" / "techdraft_manifest.json").exists()


def test_cli_rejects_unknown_step_and_has_defaults():
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])
    assert dict((l, r) for l, _, r in cli.QUALITY_RUNS) == {"quality": 0.5, "quality3b": 0.3}
    assert cli.SWEEP_ITEMS == 100
