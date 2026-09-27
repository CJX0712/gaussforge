.PHONY: install test lint demo determinism clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt ruff pytest pytest-cov

test:
	$(PY) -m pytest tests -q -W ignore::UserWarning --cov=gaussforge --cov-report=term

lint:
	$(PY) -m ruff check gaussforge tests examples

demo:
	$(PY) examples/run_demo.py

determinism:
	$(PY) -m gaussforge.cli determinism

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage
