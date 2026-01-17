SHELL := /bin/bash
PY := 3.12

.PHONY: create-venv remove-venv remove rqts_txt lock dev-install

create-venv:
	@echo "START: Creating .venv with uv (Python $(PY))" ; \
	uv venv --python $(PY) --system-site-packages ; \

setup:
	uv sync ; \
	uv run pre-commit install ; \
	uv pip install -U pip setuptools wheel ; \
	uv pip install -e .


remove:
	@echo "START: removing .venv and lock" && \
	rm -rf .venv uv.lock dist/ lit-pid-env/

rqts_txt:
	uv export --no-hashes  > requirements.txt

lock:
	uv lock

dev-install:
	uv pip install -e .
