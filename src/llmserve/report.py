"""Build figures, tables and auto-generated findings from results/processed/*.csv.

Nothing here is hand-written: every sentence in the Findings section is computed from data,
so re-running the experiments on other hardware regenerates an honest report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "results" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

ORDER = ["sequential", "static-16", "static-8", "continuous-8", "continuous-16", "continuous-32", "vllm"]
COLORS = {"sequential": "#8c8c8c", "static-16": "#d98a1c", "static-8": "#d98a1c", "continuous-8": "#2a7fbf",
          "continuous-16": "#2a7fbf", "continuous-32": "#14507a", "vllm": "#2e9e5b"}


def _read(name):
    p = PROC / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _configs(df):
    present = list(df["config"].unique())
    return [c for c in ORDER if c in present] + [c for c in present if c not in ORDER]


def _fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def md_table(df, floatfmt=2):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(_fmt(v, floatfmt) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- sections
def scheduler_section(findings):
    df = _read("scheduler_open_loop.csv")
    if df is None:
        return ""
    v = _read("vllm_open_loop.csv")
    if v is not None:
        df = pd.concat([df, v], ignore_index=True)
    cfgs = _configs(df)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in cfgs:
        g = df[df["config"] == c].sort_values("offered_rps")
        kw = dict(marker="o", label=c, color=COLORS.get(c))
        ax[0].plot(g["offered_rps"], g["output_tok_per_s"], **kw)
        ax[1].plot(g["offered_rps"], g["ttft_p95"], **kw)
        ax[2].plot(g["offered_rps"], g["goodput_rps"], **kw)
    for a in ax:
        a.set_xscale("log")
    ax[1].set_yscale("log")
    slo = df.iloc[0]
    _style(ax[0], "offered load (req/s)", "output tokens/s", "Throughput")
    _style(ax[1], "offered load (req/s)", "p95 time to first token (s)", "Tail TTFT (log scale)")
    _style(ax[2], "offered load (req/s)", "goodput (req/s)",
           f"Goodput: TTFT<={slo['slo_ttft_s']}s and TPOT<={slo['slo_tpot_s']}s")
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / "scheduler_open_loop.png", dpi=150)
    plt.close()

    # SLO capacity: highest offered rate where >=95% of requests meet both SLOs and nothing failed
    rows = []
    for c in cfgs:
        g = df[df["config"] == c].sort_values("offered_rps")
        okg = g[(g["slo_attainment"] >= 0.95) & (g["failed"] == 0)]
        cap = float(okg["offered_rps"].max()) if len(okg) else 0.0
        at = g[g["offered_rps"] == cap].iloc[0] if len(okg) else None
        rows.append({"config": c, "SLO capacity (req/s)": cap,
                     "peak output tok/s": float(g["output_tok_per_s"].max()),
                     "TTFT p95 at capacity (s)": float(at["ttft_p95"]) if at is not None else float("nan"),
                     "mean decode batch at max load": float(g.iloc[-1].get("mean_decode_batch", float("nan")))
                     if "mean_decode_batch" in g else float("nan")})
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "slo_capacity.csv", index=False)

    cap = dict(zip(tab["config"], tab["SLO capacity (req/s)"]))
    others = [c for c in cap if c != "sequential"]
    best = max(others, key=lambda c: cap[c], default=None)
    if cap.get("sequential", 0) > 0 and best:
        if cap[best] > cap["sequential"]:
            findings.append(f"Within the SLO, **{best}** sustains **{cap[best]/cap['sequential']:.1f}x** the request "
                            f"rate of sequential serving ({cap[best]:g} vs {cap['sequential']:g} req/s).")
        else:
            findings.append("No batching policy beat sequential serving on SLO capacity in the tested range "
                            "(the rate grid may be too low to saturate the server).")
    elif best and cap[best] > 0:
        findings.append("Sequential serving never met the SLO at any tested rate, while "
                        + ", ".join(f"{c} sustained {cap[c]:g} req/s" for c in others if cap[c] > 0) + ".")
    for st, co in (("static-16", "continuous-16"), ("static-8", "continuous-8")):
        if st in cap and co in cap:
            gs = df[df["config"] == st].set_index("offered_rps")
            gc = df[df["config"] == co].set_index("offered_rps")
            common = sorted(set(gs.index) & set(gc.index))
            if common:
                r = common[-1]
                a, b = gs.loc[r, "ttft_p95"], gc.loc[r, "ttft_p95"]
                if b > 0 and a == a and b == b:
                    findings.append(f"At {r:g} req/s, continuous batching cuts p95 TTFT {a/b:.1f}x versus static "
                                    f"batching of the same size ({_fmt(a)}s vs {_fmt(b)}s).")
                findings.append(f"SLO capacity: {co} {cap[co]:g} req/s vs {st} {cap[st]:g} req/s.")
            break
    if "vllm" in cap and cap["vllm"] > 0:
        ours = max((c for c in cap if c.startswith("continuous")), key=lambda c: cap[c], default=None)
        if ours and cap[ours] > 0:
            findings.append(f"Against vLLM on the same GPU, our best engine ({ours}) reaches "
                            f"{100*cap[ours]/cap['vllm']:.0f}% of vLLM's SLO capacity "
                            f"({cap[ours]:g} vs {cap['vllm']:g} req/s).")
    return ("### Scheduler comparison (open-loop Poisson arrivals)\n\n"
            "![scheduler](figures/scheduler_open_loop.png)\n\n" + md_table(tab) + "\n")


def concurrency_section(findings):
    df = _read("concurrency_closed_loop.csv")
    if df is None:
        return ""
    v = _read("vllm_closed_loop.csv")
    if v is not None:
        df = pd.concat([df, v], ignore_index=True)
    g = df.groupby(["config", "concurrency"])["output_tok_per_s"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c in _configs(df):
        s = g[g["config"] == c]
        ax.errorbar(s["concurrency"], s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                    label=c, color=COLORS.get(c))
    ax.set_xscale("log", base=2)
    _style(ax, "client concurrency", "output tokens/s (mean ± std over repeats)",
           "Closed-loop throughput vs concurrency")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "concurrency_closed_loop.png", dpi=150)
    plt.close()
    peak = g.loc[g.groupby("config")["mean"].idxmax()][["config", "concurrency", "mean", "std"]]
    peak = peak.rename(columns={"concurrency": "at concurrency", "mean": "peak tok/s (mean)", "std": "std"})
    base = g[(g["config"] == "sequential")]
    if len(base):
        b1 = float(base["mean"].max())
        for _, r in peak.iterrows():
            if r["config"] != "sequential" and b1 > 0:
                findings.append(f"Closed-loop peak: **{r['config']}** {r['peak tok/s (mean)']:.0f} tok/s "
                                f"= {r['peak tok/s (mean)']/b1:.1f}x sequential ({b1:.0f} tok/s).")
    return ("### Closed-loop throughput\n\n![closed loop](figures/concurrency_closed_loop.png)\n\n"
            + md_table(peak) + "\n")


def microbench_section(findings):
    dec, pre = _read("microbench_decode.csv"), _read("microbench_prefill.csv")
    if dec is None:
        return ""
    meta = json.loads((PROC / "microbench_meta.json").read_text()) if (PROC / "microbench_meta.json").exists() else {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(dec["batch"], dec["step_ms"], marker="o", color="#2a7fbf")
    ax[0].set_xscale("log", base=2)
    _style(ax[0], "decode batch size", "ms per decode step", f"Decode step time (ctx={int(dec['ctx'].iloc[0])})")
    ax[1].plot(dec["batch"], dec["tok_per_s"], marker="o", color="#2a7fbf")
    ax[1].set_xscale("log", base=2)
    _style(ax[1], "decode batch size", "tokens/s (engine only)", "Decode throughput")
    if pre is not None:
        ax[2].plot(pre["prompt_len"], pre["tok_per_s"], marker="o", color="#d98a1c")
        ax[2].set_xscale("log", base=2)
    _style(ax[2], "prompt length (tokens)", "prefill tokens/s", "Prefill throughput")
    plt.tight_layout()
    plt.savefig(FIGS / "microbench.png", dpi=150)
    plt.close()
    d = dec.dropna(subset=["step_ms"])
    if len(d) > 1:
        b0 = d.iloc[0]
        flat = d[d["step_ms"] <= 1.5 * b0["step_ms"]]
        kneeb = int(flat["batch"].max())
        findings.append(f"Decode step time is nearly flat in batch size: within 1.5x of batch 1 "
                        f"({_fmt(b0['step_ms'])} ms) up to batch **{kneeb}**, where decode tokens/s is "
                        f"{d[d['batch']==kneeb]['tok_per_s'].iloc[0]/b0['tok_per_s']:.1f}x batch 1. "
                        f"This is why batching raises throughput.")
        last = d.iloc[-1]
        peak_bw = meta.get("peak_bw_gbs", d["peak_bw_gbs"].iloc[0])
        util = float(last["bw_utilisation"])
        tail = ("well below the memory roofline, so the step is not bandwidth-bound; "
                "kernel-launch / Python overhead is the likely limiter (check profile_decode.txt)"
                if util < 0.3 else "approaching the memory roofline")
        findings.append(f"At batch {int(last['batch'])} a decode step takes {_fmt(last['step_ms'])} ms, "
                        f"{_fmt(100*util,1)}% of the assumed {peak_bw:g} GB/s peak bandwidth: {tail}.")
    cols = ["batch", "step_ms", "tok_per_s", "achieved_gb_per_s", "bw_utilisation"]
    return ("### Where the time goes (microbenchmarks)\n\n![microbench](figures/microbench.png)\n\n"
            + md_table(dec[cols], 3) + "\n")


def precision_section(findings):
    thr, q = _read("precision_throughput.csv"), _read("precision_quality.csv")
    if thr is None:
        return ""
    thr = thr[thr.get("error").isna()] if "error" in thr else thr
    if thr.empty:
        return ""
    agg = thr.groupby(["dtype", "quant"]).agg(tok_s=("output_tok_per_s", "mean"), std=("output_tok_per_s", "std"),
                                              gpu_mem_peak_mb=("gpu_mem_peak_mb", "max") if "gpu_mem_peak_mb" in thr else ("output_tok_per_s", "size"),
                                              ).reset_index()
    if q is not None:
        qq = q.drop(columns=[c for c in ("error", "eval_text") if c in q], errors="ignore")
        agg = agg.merge(qq, on=["dtype", "quant"], how="left")
    agg["precision"] = agg["dtype"] + "/" + agg["quant"]
    agg.to_csv(PROC / "precision_summary.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].bar(agg["precision"], agg["tok_s"], yerr=agg["std"].fillna(0), color="#2a7fbf", capsize=3)
    _style(ax[0], "", "output tokens/s", "Throughput")
    if "gpu_mem_peak_mb" in agg:
        ax[1].bar(agg["precision"], agg["gpu_mem_peak_mb"], color="#d98a1c")
    _style(ax[1], "", "peak GPU memory (MB)", "Memory")
    if "perplexity" in agg:
        ax[2].bar(agg["precision"], agg["perplexity"], color="#2e9e5b")
    _style(ax[2], "", "perplexity (lower is better)", "Quality")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIGS / "precision.png", dpi=150)
    plt.close()
    ref = agg[(agg["dtype"] == "float32")]
    if len(ref):
        r = ref.iloc[0]
        for _, x in agg.iterrows():
            if x["precision"] == r["precision"]:
                continue
            s = f"{x['precision']}: {x['tok_s']/r['tok_s']:.2f}x throughput vs fp32"
            if "perplexity" in agg and x.get("perplexity") == x.get("perplexity") and r.get("perplexity") == r.get("perplexity"):
                s += f", perplexity {_fmt(x['perplexity'],3)} vs {_fmt(r['perplexity'],3)}"
            if "greedy_token_agreement" in agg and x.get("greedy_token_agreement") == x.get("greedy_token_agreement"):
                s += f", {100*x['greedy_token_agreement']:.0f}% greedy-token agreement"
            findings.append(s + ".")
    show = agg.drop(columns=["precision"]).round(3)
    return "### Precision / quantization trade-off\n\n![precision](figures/precision.png)\n\n" + md_table(show, 3) + "\n"


def build():
    FIGS.mkdir(parents=True, exist_ok=True)
    findings = []
    env = PROC / "environment.json"
    envd = json.loads(env.read_text()) if env.exists() else {}
    sections = []
    for fn in (scheduler_section, concurrency_section, microbench_section, precision_section):
        try:
            sections.append(fn(findings))
        except Exception as exc:  # a missing experiment must not break the report
            sections.append(f"*(section {fn.__name__} skipped: {type(exc).__name__}: {exc})*\n")
    hw = envd.get("gpu") or f"CPU ({envd.get('cpu_count', '?')} cores)"
    head = (f"**Setup:** model `{envd.get('profile_model', '?')}`, {envd.get('dtype', '?')}, hardware: {hw}, "
            f"torch {envd.get('torch', '?')}, transformers {envd.get('transformers', '?')}, "
            f"run date {envd.get('date_utc', '?')} UTC, commit `{envd.get('git_commit', '?')}`.\n")
    fnd = "### Key findings (auto-generated from the data)\n\n" + \
          ("\n".join(f"- {f}" for f in findings) if findings else "- (no results yet)") + "\n"
    body = head + "\n" + fnd + "\n" + "\n".join(s for s in sections if s)
    (REPORTS / "RESULTS.md").write_text("# Results\n\n" + body, encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        compact = (head + "\n" + fnd + "\n" +
                   "".join(f"![{n}](reports/figures/{n}.png)\n\n" for n in
                           ("scheduler_open_loop", "concurrency_closed_loop", "microbench", "precision")
                           if (FIGS / f"{n}.png").exists()) +
                   "Full tables and methodology: [reports/RESULTS.md](reports/RESULTS.md).\n")
        if START in txt and END in txt:
            txt = txt[:txt.index(START) + len(START)] + "\n" + compact + txt[txt.index(END):]
            readme.write_text(txt, encoding="utf-8")
    print("\n".join(findings) or "(no findings)")
    return findings


if __name__ == "__main__":
    build()
