# neuralSpring — Makefile (fallback for systems without just)
# Prefer: just check
# Usage:  make check    — all quality gates
#         make test     — Python + Rust tests
#         make lint     — lint + format check
#         make baselines — full Python suite (~6 min)

.PHONY: check lint test validate baselines lint-python lint-rust test-python test-rust fix fmt

check: lint test validate
	@echo ""
	@echo "━━━ ALL GATES PASS ━━━"

test: test-python test-rust

lint: lint-python lint-rust

lint-python:
	ruff check control/ scripts/ tests/
	ruff format --check control/ tests/

lint-rust:
	cargo clippy -- -D warnings
	cargo fmt --check
	cargo doc --no-deps

test-python:
	python3 -m pytest tests/ -v --tb=short

test-rust:
	cargo test

validate:
	cargo run --bin validate_surrogate
	cargo run --bin validate_transformer
	cargo run --bin validate_metrics

baselines:
	bash scripts/run_all_baselines.sh

fix:
	ruff check --fix control/ scripts/ tests/
	ruff format control/ tests/

fmt:
	ruff format control/ tests/
	cargo fmt
