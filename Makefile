install:
	pip install -e .

test:
	python -m pytest -q

smoke:   # whole pipeline on CPU with a tiny random model (~5 min)
	python -m llmserve.experiments --profile smoke --only env scheduler concurrency microbench precision decode_graphs engines chunked prefix
	python -m llmserve.report

gpu-test:   # includes the CUDA-graph correctness test
	python -m pytest -q tests/test_slot_engine.py

t4:      # the real experiments (GPU). See docs/GPU_BENCHMARKS.md
	python -m llmserve.experiments --profile t4 --only env microbench decode_graphs scheduler concurrency ablation precision quality engines chunked prefix scale
	python -m llmserve.report

.PHONY: install test smoke gpu-test t4
