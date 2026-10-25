install:
	pip install -e .

test:
	python -m pytest -q

smoke:   # whole pipeline on CPU with a tiny random model (~3 min)
	python -m llmserve.experiments --profile smoke --only env scheduler concurrency microbench precision
	python -m llmserve.report

t4:      # the real experiments (GPU)
	python -m llmserve.experiments --profile t4 --only env microbench scheduler concurrency precision quality
	python -m llmserve.report

.PHONY: install test smoke t4
