.PHONY: install train quantize-all bench-all evaluate-all report pipeline test test-int lint format typecheck clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
EXPLORER ?= .venv/bin/quant-explorer

install:
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
	$(PIP) install -e ".[dev]"

train:
	$(EXPLORER) train --epochs 5 --batch-size 128

quantize-all:
	$(EXPLORER) quantize --config dynamic_int8
	$(EXPLORER) quantize --config static_int8_per_tensor
	$(EXPLORER) quantize --config static_int8_per_channel

bench-all:
	$(EXPLORER) bench --config fp32_baseline
	$(EXPLORER) bench --config dynamic_int8
	$(EXPLORER) bench --config static_int8_per_tensor
	$(EXPLORER) bench --config static_int8_per_channel

evaluate-all:
	$(EXPLORER) evaluate --config fp32_baseline
	$(EXPLORER) evaluate --config dynamic_int8
	$(EXPLORER) evaluate --config static_int8_per_tensor
	$(EXPLORER) evaluate --config static_int8_per_channel

report:
	$(EXPLORER) report

pipeline: train quantize-all bench-all evaluate-all report

pipeline-tiny:
	$(EXPLORER) pipeline --tiny

test:
	$(PY) -m pytest tests/unit

test-int:
	RUN_INTEGRATION=1 $(PY) -m pytest tests/integration

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

typecheck:
	$(PY) -m mypy src/quant_explorer

clean:
	rm -rf data/cifar-10-batches-py data/cifar-10-python.tar.gz
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
