# Contributor entry points. Each target wraps one existing command or script;
# scripts/check.sh stays the gate and does not depend on make (the weekly
# Windows CI cell has no GNU make on PATH).
.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck hooks install-hooks test test-fast check check-fast docs

help: ## List the targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Locked dev environment (what CI installs)
	uv sync --locked --group dev

lint: ## ruff format check and lint (check.sh step 2)
	uv run ruff format --check .
	uv run ruff check .

format: ## ruff: rewrite in place
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## mypy on the package (check.sh step 2)
	uv run mypy src

hooks: ## Every hook in .pre-commit-config.yaml on every file (prek)
	uv run prek run --all-files

install-hooks: ## Install the prek git hook
	uv run prek install

test: ## Implementer tests, both dtype modes
	scripts/run_tests.sh

test-fast: ## Implementer tests without the slow list
	scripts/run_tests.sh --fast

check: ## Full gate: guard, static checks, tests, doctests, spec tests
	scripts/check.sh

check-fast: ## Step gate: check without the slow list
	scripts/check.sh --fast

docs: ## Build the docs (needs the docs and examples groups)
	scripts/build_docs.sh
