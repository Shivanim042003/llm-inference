# GPU benchmark procedure

CI runs only the CPU correctness tests (`.github/workflows/tests.yml`). Performance numbers come from a manual GPU run:

1. Use a single GPU class per comparison (Colab T4 for the published numbers). Do not mix sessions.
2. `pip install -r requirements-lock.txt` (+ a matching CUDA torch), then `pip install -e . --no-deps`.
3. `make gpu-test` must pass first: it checks the CUDA-graph decode against the eager path.
4. `make t4` runs every experiment; the vLLM baseline is run separately in its own virtualenv
   (`python -m llmserve.experiments --profile t4 --only vllm`) and the report states whether it completed.
5. `python -m llmserve.report` regenerates `reports/RESULTS.md` and the README block. Quote numbers only from there.
6. The load generator shares the machine with the server; if CPU is saturated, treat throughput as client-bound.

Run-to-run variation: open-loop points use several seeds, closed-loop points use repeats; the report shows mean ± sd.
Compare numbers only within a run (shared GPUs differ between sessions).
