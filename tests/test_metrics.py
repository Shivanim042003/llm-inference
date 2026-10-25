import math

from llmserve.metrics import RequestRecord, bootstrap_ci, percentile, summarize


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_summarize_goodput_and_failures():
    recs = [
        RequestRecord(0, True, ttft=0.1, e2e=1.1, output_tokens=11),   # tpot 0.1
        RequestRecord(0, True, ttft=2.0, e2e=3.0, output_tokens=11),   # violates ttft slo
        RequestRecord(0, False, error="timeout"),
    ]
    s = summarize(recs, wall_s=10.0, slo_ttft=1.0, slo_tpot=0.11)
    assert s["succeeded"] == 2 and s["failed"] == 1
    assert abs(s["throughput_rps"] - 0.2) < 1e-9
    assert abs(s["slo_attainment"] - 1 / 3) < 1e-9
    assert abs(s["goodput_rps"] - 0.1) < 1e-9
    assert abs(s["output_tok_per_s"] - 2.2) < 1e-9


def test_tpot_needs_two_tokens():
    assert RequestRecord(0, True, ttft=0.1, e2e=0.2, output_tokens=1).tpot is None


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo <= 3.0 <= hi
