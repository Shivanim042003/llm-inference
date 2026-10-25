import pandas as pd

from llmserve import report


def test_report_builds_from_synthetic_results(tmp_path, monkeypatch):
    proc, reports = tmp_path / "processed", tmp_path / "reports"
    proc.mkdir()
    monkeypatch.setattr(report, "PROC", proc)
    monkeypatch.setattr(report, "REPORTS", reports)
    monkeypatch.setattr(report, "FIGS", reports / "figures")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    rows = []
    for cfg, mult in [("sequential", 1.0), ("continuous-16", 4.0)]:
        for rate in [0.5, 1, 2, 4]:
            ok = rate <= 0.5 * mult
            rows.append(dict(config=cfg, offered_rps=rate, failed=0, slo_attainment=1.0 if ok else 0.2,
                             output_tok_per_s=30 * mult * min(rate, 0.5 * mult), ttft_p95=0.3 if ok else 9.0,
                             goodput_rps=rate if ok else 0.1, slo_ttft_s=1.0, slo_tpot_s=0.1,
                             mean_decode_batch=1 + mult))
    pd.DataFrame(rows).to_csv(proc / "scheduler_open_loop.csv", index=False)
    (tmp_path / "README.md").write_text(f"x\n{report.START}\nold\n{report.END}\ny\n")
    findings = report.build()
    assert any("sequential" in f for f in findings)
    assert (reports / "RESULTS.md").exists()
    assert "old" not in (tmp_path / "README.md").read_text()
