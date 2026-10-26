# TechDraft: evidence-first technical-support answering on real forum questions

Data: the public TechQA questions (real IBM developer-forum posts) and the IBM technote documents they refer to, from the Hugging Face copy `nvidia/TechQA-RAG-Eval`. Raw data are **not** in this repository. Labels: **MEASURED** = observed in the run; **AUTOMATED** = rule-based overlap check, not a human judgment; no cost saving or productivity gain was measured.

## Setup and what this is not

- 910 questions (300 unanswerable) and 496 documents (2457 chunks of about 180 words). Split by hash of the question id: `tune` (347 questions, used for prompt and threshold choices) and `final` (560, evaluated after freezing).
- This is **our own split of a merged copy**, not the official train/dev split. The knowledge base is built from the documents of *all* questions, so each question's correct document is always in the index, and unanswerable questions have only other questions' documents to match. That makes retrieval easier than a real deployment and makes abstention a test of whether the retrieved text answers the question.
- 'Pass' for an answerable question means: the model answered, every quoted sentence appears verbatim in the excerpt it cites, and the answer's token F1 against the gold answer is at least 0.2. For an unanswerable question it means the model abstained. This is an overlap check, **not** a measure of whether a customer would find the reply useful.
- 3 question(s) were excluded because the prompt would not fit the 4,096-token server limit with 300 generated tokens; they are not truncated or scored. Excluded ids: `tune` ['TRAIN_Q063'], `final` ['TRAIN_Q090', 'DEV_Q010']. Every model and baseline below is computed on exactly the remaining ids; hashes of those id sets, the prompt and the parser are in `results/processed/techdraft_manifest.json`.
- Unanswerable here means 'the provided documents do not contain the answer', not 'a human must act'.

## 1. Retrieval (MEASURED, backend-independent)

| split | questions | unanswerable | gold document in top-4 |
|---|---|---|---|
| tune | 347 | 122 | 87% |
| final | 560 | 178 | 88% |

## 2. Answer quality (AUTOMATED, `final` split, one pass over every question)

| system | pass | answerable: pass | unanswerable: abstained | answer F1 (0 if abstained) | evidence recall | quotes grounded | abstain precision | abstain recall | unparsed |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B / slot-graph-32 | 19% | 22% | 11% | 0.20 | 21% | 54% | 30% | 26% | 22% |
| Qwen2.5-3B / slot-graph-32 | 34% | 7% | 93% | 0.05 | 34% | 64% | 38% | 93% | 0% |
| baseline: always abstain | 32% | 0% | 100% | 0.00 | n/a | n/a | 32% | 100% | 0% |
| baseline: quote top-1 chunk (no LLM) | 34% | 50% | 0% | 0.28 | 68% | 100% | n/a | 0% | 0% |
| baseline: top-1 chunk + score threshold (no LLM) | 42% | 20% | 87% | 0.11 | 81% | 100% | 37% | 87% | 0% |


The same measures on the `tune` split (the split used for choices):

| system | pass | answerable: pass | unanswerable: abstained | answer F1 (0 if abstained) | evidence recall | quotes grounded | abstain precision | abstain recall | unparsed |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B / slot-graph-32 | 16% | 17% | 12% | 0.18 | 23% | 54% | 30% | 21% | 18% |
| Qwen2.5-3B / slot-graph-32 | 38% | 9% | 93% | 0.06 | 41% | 72% | 42% | 93% | 0% |
| baseline: always abstain | 35% | 0% | 100% | 0.00 | n/a | n/a | 35% | 100% | 0% |
| baseline: quote top-1 chunk (no LLM) | 31% | 48% | 0% | 0.25 | 69% | 100% | n/a | 0% | 0% |
| baseline: top-1 chunk + score threshold (no LLM) | 43% | 19% | 88% | 0.10 | 76% | 100% | 42% | 88% | 0% |


## 3. Serving under load (MEASURED, `final` split, sampled questions)

Model served in the sweep: `Qwen/Qwen2.5-1.5B-Instruct`.

| backend | offered/s | requests | failed | p95 TTFT (s) | p95 draft time (s) | on time | passing & on time |
|---|---|---|---|---|---|---|---|
| dynamic-32 | 0.25 | 100 | 0 | 1.93 | 14.63 | 95% | 19% |
| dynamic-32 | 0.50 | 100 | 0 | 2.29 | 40.70 | 48% | 11% |
| dynamic-32 | 0.75 | 100 | 5 | 44.96 | 106.44 | 3% | 2% |
| dynamic-32 | 1.00 | 100 | 8 | 71.44 | 114.69 | 1% | 1% |
| dynamic-32 | 1.50 | 100 | 29 | 95.11 | 115.31 | 0% | 0% |
| slot-graph-32 | 0.25 | 100 | 0 | 1.88 | 9.75 | 100% | 19% |
| slot-graph-32 | 0.50 | 100 | 0 | 2.10 | 13.23 | 98% | 19% |
| slot-graph-32 | 0.75 | 100 | 0 | 2.42 | 21.15 | 82% | 19% |
| slot-graph-32 | 1.00 | 100 | 0 | 3.42 | 51.76 | 33% | 7% |
| slot-graph-32 | 1.50 | 100 | 0 | 26.67 | 62.58 | 3% | 1% |
| slot-graph-32+prefix | 0.25 | 100 | 0 | 0.70 | 7.40 | 100% | 19% |
| slot-graph-32+prefix | 0.50 | 100 | 0 | 0.80 | 9.01 | 100% | 19% |
| slot-graph-32+prefix | 0.75 | 100 | 0 | 0.82 | 10.11 | 100% | 18% |
| slot-graph-32+prefix | 1.00 | 100 | 0 | 0.91 | 11.32 | 99% | 20% |
| slot-graph-32+prefix | 1.50 | 100 | 0 | 1.15 | 20.21 | 83% | 19% |


Sustainable rate = highest tested offered rate with zero failures and at least 95% of drafts on time:

| backend | sustainable_rate | next_tested_rate |
|---|---|---|
| dynamic-32 | 0.25 | 0.50 |
| slot-graph-32 | 0.50 | 0.75 |
| slot-graph-32+prefix | 1.00 | 1.50 |


## Limits

- One model, one GPU, greedy decoding, Poisson arrivals; the evaluation set is a few hundred questions, so a few points of difference are noise.
- The split is ours, on a merged copy; the final set was evaluated after freezing the pipeline, but any later tuning against it would remove that property.
- Overlap with a gold answer rewards copying; the top-chunk baselines show how much of the score a no-LLM system gets.
- The pass rule checks that quoted sentences appear in the excerpts and that the answer overlaps the gold answer; it does not check that every statement in the answer follows from the quotes.
- No human judged these replies. Prompts are long (retrieved technote excerpts), so serving results differ from short-prompt workloads.
