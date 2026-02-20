# neuralSpring — Makefile (fallback for systems without just)
# Prefer: just check
# Usage:  make check    — all quality gates
#         make test     — Python + Rust tests
#         make lint     — lint + format check
#         make baselines — full Python suite (~6 min)

.PHONY: check lint test validate validate-native validate-barracuda validate-ml validate-tensor-cpu validate-tensor-gpu validate-tensor-all bench-tensor bench-tensor-compare bench-ml bench-fused baselines lint-python lint-rust test-python test-rust fix fmt coverage

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

validate: validate-native validate-barracuda

validate-native:
	cargo run --bin validate_surrogate
	cargo run --bin validate_transformer
	cargo run --bin validate_metrics

validate-barracuda:
	cargo run --bin validate_barracuda_stats
	cargo run --bin validate_barracuda_linalg
	cargo run --bin validate_barracuda_special
	cargo run --bin validate_barracuda_optimize
	cargo run --bin validate_barracuda_precision
	NEURALSPRING_BACKEND=cpu cargo run --bin validate_barracuda_tensor
	NEURALSPRING_BACKEND=gpu cargo run --bin validate_barracuda_tensor
	cargo run --bin validate_barracuda_tensor_f64
	cargo run --bin validate_barracuda_quantized
	cargo run --bin validate_barracuda_linalg_ext
	cargo run --bin validate_barracuda_ml_inference

validate-tensor-cpu:
	NEURALSPRING_BACKEND=cpu cargo run --bin validate_barracuda_tensor

validate-tensor-gpu:
	NEURALSPRING_BACKEND=gpu cargo run --bin validate_barracuda_tensor

validate-tensor-all:
	@echo "── Tensor: CPU (llvmpipe) ──"
	NEURALSPRING_BACKEND=cpu cargo run --bin validate_barracuda_tensor
	@echo "── Tensor: GPU ──"
	NEURALSPRING_BACKEND=gpu cargo run --bin validate_barracuda_tensor
	@echo "── WGSL math universal across hardware ──"

validate-ml:
	cargo run --bin validate_barracuda_ml_inference

bench-tensor:
	cargo run --release --bin bench_barracuda_tensor

bench-tensor-compare:
	@echo "── Benchmark: CPU (llvmpipe) ──"
	NEURALSPRING_BACKEND=cpu cargo run --release --bin bench_barracuda_tensor
	@echo ""
	@echo "── Benchmark: GPU ──"
	NEURALSPRING_BACKEND=gpu cargo run --release --bin bench_barracuda_tensor

bench-ml:
	cargo run --release --bin bench_mlp_inference
	cargo run --release --bin bench_transformer_block

bench-fused:
	@echo "── Fused Pipeline: CPU ──"
	NEURALSPRING_BACKEND=cpu cargo run --release --bin bench_fused_inference
	@echo ""
	@echo "── Fused Pipeline: GPU ──"
	NEURALSPRING_BACKEND=gpu cargo run --release --bin bench_fused_inference

baselines:
	bash scripts/run_all_baselines.sh

coverage:
	cargo llvm-cov --lib --html
	@echo "Coverage report: target/llvm-cov/html/index.html"

coverage-json:
	cargo llvm-cov --lib --json --output-path target/llvm-cov/coverage.json

fix:
	ruff check --fix control/ scripts/ tests/
	ruff format control/ tests/

fmt:
	ruff format control/ tests/
	cargo fmt
