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


def test_report_flags_censored_vllm_and_builds_ablation(tmp_path, monkeypatch):
    proc, reports = tmp_path / "processed", tmp_path / "reports"
    proc.mkdir()
    for name in ("PROC", "REPORTS", "FIGS", "ROOT"):
        monkeypatch.setattr(report, name, {"PROC": proc, "REPORTS": reports, "FIGS": reports / "figures", "ROOT": tmp_path}[name])

    def rows(cfg, rates, ok_upto, mult):
        out = []
        for r in rates:
            ok = r <= ok_upto
            out.append(dict(config=cfg, offered_rps=r, failed=0, slo_attainment=1.0 if ok else 0.1,
                            output_tok_per_s=50 * mult, ttft_p95=0.1 if ok else 20.0, goodput_rps=r if ok else 0.1,
                            slo_ttft_s=1.0, slo_tpot_s=0.1, mean_decode_batch=4))
        return out
    pd.DataFrame(rows("sequential", [1, 2, 4], 0, 1) + rows("static-16", [1, 2, 4], 1, 2)
                 + rows("continuous-16", [1, 2, 4], 2, 4)).to_csv(proc / "scheduler_open_loop.csv", index=False)
    pd.DataFrame(rows("vllm", [1, 2, 4], 99, 10)).to_csv(proc / "vllm_open_loop.csv", index=False)
    ab = []
    for v, tps in (("full-logits", 100.0), ("last-logits", 130.0)):
        for rate in (1, 2):
            ab.append(dict(variant=v, kind="open_loop", offered_rps=rate, output_tok_per_s=tps, ttft_p95=1.0))
        ab.append(dict(variant=v, kind="closed_loop", output_tok_per_s=tps, prefill_time_share=0.3 if v == "full-logits" else 0.1))
    pd.DataFrame(ab).to_csv(proc / "ablation_logits.csv", index=False)
    findings = report.build()
    text = " ".join(findings)
    assert "at most" in text            # vLLM never saturated -> ratio is an upper bound
    assert "1.30x" in text and "raises" in text   # ablation speed-up computed from data
    assert "static" in text and "lower p95 TTFT" in text
