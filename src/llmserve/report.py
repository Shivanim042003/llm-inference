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

    # SLO capacity: highest offered rate where >=95% of requests meet both SLOs and nothing failed.
    # It is only known to lie between that rate and the next tested rate; if it equals the highest
    # tested rate the configuration never saturated and the capacity is a lower bound (">=").
    rows, cap, upper = [], {}, {}
    for c in cfgs:
        g = df[df["config"] == c].sort_values("offered_rps")
        okg = g[(g["slo_attainment"] >= 0.95) & (g["failed"] == 0)]
        capv = float(okg["offered_rps"].max()) if len(okg) else 0.0
        higher = g[g["offered_rps"] > capv]["offered_rps"]
        up = float(higher.min()) if len(higher) else float("inf")
        cap[c], upper[c] = capv, up
        at = g[g["offered_rps"] == capv].iloc[0] if len(okg) else None
        label = (f">= {capv:g} (never saturated)" if up == float("inf") and capv > 0
                 else f"{capv:g} (true value in [{capv:g}, {up:g}))" if capv > 0 else "0 (SLO never met)")
        rows.append({"config": c, "SLO capacity (req/s)": label,
                     "peak output tok/s": float(g["output_tok_per_s"].max()),
                     "TTFT p95 at capacity (s)": float(at["ttft_p95"]) if at is not None else float("nan"),
                     "mean decode batch at max load": float(g.iloc[-1]["mean_decode_batch"])
                     if "mean_decode_batch" in g and g.iloc[-1]["mean_decode_batch"] == g.iloc[-1]["mean_decode_batch"] else float("nan")})
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "slo_capacity.csv", index=False)

    others = [c for c in cap if c != "sequential"]
    best = max(others, key=lambda c: cap[c], default=None)
    if cap.get("sequential", 0) > 0 and best:
        if cap[best] > cap["sequential"]:
            findings.append(f"Within the SLO, **{best}** sustains at least {cap[best]/cap['sequential']:.1f}x the request "
                            f"rate of sequential serving ({cap[best]:g} vs {cap['sequential']:g} req/s).")
        else:
            findings.append("No batching policy beat sequential serving on SLO capacity in the tested range "
                            "(the rate grid may be too low to saturate the server).")
    elif best and cap[best] > 0:
        findings.append("Sequential serving never met the SLO at any tested rate, while "
                        + ", ".join(f"{c} sustained {cap[c]:g} req/s" for c in others if cap[c] > 0) + ".")
    # continuous vs static at the highest rate where continuous still meets the SLO (not in overload)
    for st, co in (("static-16", "continuous-16"), ("static-8", "continuous-8")):
        if st in cap and co in cap and cap[co] > 0:
            gs = df[df["config"] == st].set_index("offered_rps")
            gc = df[df["config"] == co].set_index("offered_rps")
            r = cap[co]
            if r in gs.index and r in gc.index:
                x, y = gs.loc[r, "ttft_p95"], gc.loc[r, "ttft_p95"]
                if y > 0 and x == x and y == y:
                    findings.append(f"At {r:g} req/s (the highest rate where {co} meets the SLO), continuous batching has "
                                    f"{x/y:.0f}x lower p95 TTFT than static batching of the same size "
                                    f"({_fmt(y)}s vs {_fmt(x)}s). Static batching meets the SLO at "
                                    f"{'no tested rate' if cap[st] == 0 else f'up to {cap[st]:g} req/s'}.")
            break
    if "vllm" in cap and cap["vllm"] > 0:
        ours = max((c for c in cap if c.startswith("continuous")), key=lambda c: cap[c], default=None)
        if ours and cap[ours] > 0:
            ratio = cap[ours] / cap["vllm"]
            censored = upper["vllm"] == float("inf")
            findings.append(
                f"Against vLLM on the same GPU, our best engine ({ours}, SLO capacity {cap[ours]:g} req/s) reaches "
                + (f"at most {100*ratio:.0f}% of vLLM's SLO capacity (vLLM met the SLO at every rate tested, up to "
                   f"{cap['vllm']:g} req/s, so its true capacity is higher)" if censored
                   else f"{100*ratio:.0f}% of vLLM's SLO capacity ({cap['vllm']:g} req/s)") + ".")
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
        for variant, gp in (pre.groupby("variant") if "variant" in pre else [("prefill", pre)]):
            ax[2].plot(gp["prompt_len"], gp["tok_per_s"], marker="o", label=variant)
        ax[2].legend(frameon=False, fontsize=8)
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
    if pre is not None and "variant" in pre:
        pv = pre.pivot_table(index="prompt_len", columns="variant", values="seconds")
        if {"full_logits", "last_logits"} <= set(pv.columns):
            big = pv.index.max()
            findings.append(f"Prefill of a {int(big)}-token prompt takes {1e3*pv.loc[big,'last_logits']:.1f} ms when only the "
                            f"last position's logits are computed vs {1e3*pv.loc[big,'full_logits']:.1f} ms with full-vocab logits "
                            f"at every position ({pv.loc[big,'full_logits']/pv.loc[big,'last_logits']:.1f}x).")
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
            if "greedy_prefix_match" in agg and x.get("greedy_prefix_match") == x.get("greedy_prefix_match"):
                s += f", greedy outputs match fp32 for the first {100*x['greedy_prefix_match']:.0f}% of tokens on average"
            findings.append(s + ".")
    note = ""
    if q is not None and "eval_text" in q and q["eval_text"].astype(str).str.contains("NOT wikitext|fallback", case=False).any():
        findings.append("WARNING: perplexity was measured on a built-in repeated paragraph, not WikiText-2; "
                        "values are only comparable between rows of this table.")
        note = "\n*Perplexity here uses fallback text (WikiText-2 was unavailable); compare rows only with each other.*\n"
    show = agg.drop(columns=["precision"]).round(3)
    return ("### Precision / quantization trade-off\n\n![precision](figures/precision.png)\n\n"
            + md_table(show, 3) + "\n" + note)


def ablation_section(findings):
    df = _read("ablation_logits.csv")
    if df is None:
        return ""
    cl = df[df["kind"] == "closed_loop"]
    ol = df[df["kind"] == "open_loop"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for variant, color in (("full-logits", "#8c8c8c"), ("last-logits", "#2a7fbf")):
        g = ol[ol["variant"] == variant].sort_values("offered_rps")
        ax[0].plot(g["offered_rps"], g["output_tok_per_s"], marker="o", label=variant, color=color)
        ax[1].plot(g["offered_rps"], g["ttft_p95"], marker="o", label=variant, color=color)
    m = cl.groupby("variant")[["output_tok_per_s", "prefill_time_share"]].mean()
    if len(m):
        ax[2].bar(m.index, m["output_tok_per_s"], color=["#8c8c8c" if v == "full-logits" else "#2a7fbf" for v in m.index])
    ax[1].set_yscale("log")
    _style(ax[0], "offered load (req/s)", "output tokens/s", "Open-loop throughput")
    _style(ax[1], "offered load (req/s)", "p95 TTFT (s, log)", "Tail TTFT")
    _style(ax[2], "", "output tokens/s", "Closed-loop peak (same concurrency)")
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "ablation_logits.png", dpi=150)
    plt.close()
    if {"full-logits", "last-logits"} <= set(m.index):
        a, b = m.loc["full-logits"], m.loc["last-logits"]
        ratio = b["output_tok_per_s"] / a["output_tok_per_s"]
        verb = ("raises" if ratio >= 1.03 else "lowers" if ratio <= 0.97 else "does not measurably change")
        findings.append(f"Computing the lm_head only for the last prompt position {verb} closed-loop throughput "
                        f"({a['output_tok_per_s']:.0f} -> {b['output_tok_per_s']:.0f} tok/s, {ratio:.2f}x; "
                        f"run-to-run spread is not shown, so treat differences under ~5% as noise) "
                        f"and moves the share of GPU time spent in prefill from {100*a['prefill_time_share']:.0f}% to "
                        f"{100*b['prefill_time_share']:.0f}%.")
    show = m.reset_index().rename(columns={"output_tok_per_s": "closed-loop tok/s", "prefill_time_share": "prefill time share"})
    return ("### Optimisation ablation: last-position logits in prefill\n\n![ablation](figures/ablation_logits.png)\n\n"
            + md_table(show, 3) + "\n")


def build():
    FIGS.mkdir(parents=True, exist_ok=True)
    findings = []
    env = PROC / "environment.json"
    envd = json.loads(env.read_text()) if env.exists() else {}
    sections = []
    for fn in (scheduler_section, concurrency_section, microbench_section, ablation_section, precision_section):
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
                           ("scheduler_open_loop", "concurrency_closed_loop", "microbench", "ablation_logits", "precision")
                           if (FIGS / f"{n}.png").exists()) +
                   "Full tables and methodology: [reports/RESULTS.md](reports/RESULTS.md).\n")
        if START in txt and END in txt:
            txt = txt[:txt.index(START) + len(START)] + "\n" + compact + txt[txt.index(END):]
            readme.write_text(txt, encoding="utf-8")
    print("\n".join(findings) or "(no findings)")
    return findings


if __name__ == "__main__":
    build()
