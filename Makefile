# Thin convenience wrapper around uv. Deliberately not a build system:
# every target is a one-line call to uv, and dependency logic lives only in
# pyproject.toml and uv.lock. If a target grows real logic, it belongs in
# scripts/ or in the package.

.DEFAULT_GOAL := help
.PHONY: help setup setup-inference doctor lint format format-check typecheck \
        test test-fast quality verify colab-requirements lock hooks clean

UV ?= uv

help:  ## Show this help
	@echo "panaf-ape-detection -- available targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Typical first run:  make setup && make doctor && make quality"

setup:  ## Install the lightweight development environment
	$(UV) sync

setup-inference:  ## Install the heavy inference stack (PyTorch, PyTorch-Wildlife, OpenCV)
	$(UV) sync --extra inference

doctor:  ## Report the environment a pipeline run would execute in
	$(UV) run panaf-phase1 doctor

lint:  ## Run Ruff lint checks
	$(UV) run ruff check .

format:  ## Reformat the codebase with Ruff
	$(UV) run ruff format .

format-check:  ## Check formatting without modifying files
	$(UV) run ruff format --check .

typecheck:  ## Run mypy over src/
	$(UV) run mypy src

test:  ## Run the test suite with coverage
	$(UV) run pytest

test-fast:  ## Run the test suite without coverage
	$(UV) run pytest --no-cov -q

quality: lint format-check typecheck test  ## Run every quality gate (what CI runs)

verify:  ## Verify repository structural invariants
	$(UV) run python scripts/verify_repository.py

colab-requirements:  ## Regenerate requirements-colab.txt from uv.lock
	$(UV) export --extra inference --no-hashes --no-dev \
		--format requirements-txt -o requirements-colab.txt

lock:  ## Re-resolve uv.lock, then refresh the Colab export
	$(UV) lock
	$(MAKE) colab-requirements

hooks:  ## Install pre-commit hooks
	$(UV) run pre-commit install

clean:  ## Remove caches and build output (never touches data/ or artifacts/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} +
	@echo "Caches cleared. data/ and artifacts/ were not touched."
