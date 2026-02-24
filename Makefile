# neuralSpring — Makefile (fallback for systems without just)
# Prefer: just check
# Usage:  make check    — all quality gates
#         make test     — Python + Rust tests
#         make lint     — lint + format check
#         make baselines — full Python suite (~6 min)

.PHONY: check lint test validate validate-native validate-native-papers validate-all validate-barracuda validate-barracuda-cpu validate-dispatch validate-ml validate-tensor-cpu validate-tensor-gpu validate-tensor-all bench-tensor bench-tensor-compare bench-ml bench-fused baselines lint-python lint-rust test-python test-rust fix fmt coverage

check: lint test validate
	@echo ""
	@echo "━━━ ALL GATES PASS ━━━"

test: test-python test-rust

lint: lint-python lint-rust

lint-python:
	ruff check control/ scripts/ tests/
	ruff format --check control/ tests/

lint-rust:
	cargo clippy --all-targets -- -D warnings -W clippy::pedantic -W clippy::nursery
	cargo fmt --check
	cargo doc --no-deps

test-python:
	python3 -m pytest tests/ -v --tb=short

test-rust:
	cargo test

validate: validate-all

validate-native:
	cargo run --bin validate_surrogate
	cargo run --bin validate_transformer
	cargo run --bin validate_metrics

validate-native-papers:
	cargo run --bin validate_counterdiabatic
	cargo run --bin validate_modes
	cargo run --bin validate_eco_dynamics
	cargo run --bin validate_directed_evolution
	cargo run --bin validate_hmm
	cargo run --bin validate_game_theory
	cargo run --bin validate_regulatory_network
	cargo run --bin validate_signal_integration
	cargo run --bin validate_swarm_robotics
	cargo run --bin validate_sate_alignment
	cargo run --bin validate_introgression
	cargo run --bin validate_spectral_commutativity
	cargo run --bin validate_anderson_localization
	cargo run --bin validate_pangenome_selection
	cargo run --bin validate_meta_population
	cargo run --bin validate_sequence
	cargo run --bin validate_pinn
	cargo run --bin validate_deeponet

validate-dispatch:
	cargo run --bin validate_compute_dispatch
	cargo run --bin validate_mixed_hardware
	cargo run --bin validate_basecamp_dispatch
	cargo run --bin validate_barracuda_parity
	cargo run --bin validate_metalforge_pcie

validate-all: validate-native validate-native-papers validate-barracuda validate-barracuda-cpu validate-dispatch
	@echo "━━━ All validation binaries PASS ━━━"

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

validate-barracuda-cpu:
	cargo run --bin validate_barracuda_spectral
	cargo run --bin validate_barracuda_anderson
	cargo run --bin validate_barracuda_regulatory
	cargo run --bin validate_barracuda_signal
	cargo run --bin validate_barracuda_hmm
	cargo run --bin validate_barracuda_introgression
	cargo run --bin validate_barracuda_counterdiabatic
	cargo run --bin validate_barracuda_modes
	cargo run --bin validate_barracuda_eco
	cargo run --bin validate_barracuda_directed
	cargo run --bin validate_barracuda_swarm
	cargo run --bin validate_barracuda_sate
	cargo run --bin validate_barracuda_game
	cargo run --bin validate_barracuda_pangenome
	cargo run --bin validate_barracuda_meta_pop
	cargo run --bin validate_barracuda_pinn
	cargo run --bin validate_barracuda_deeponet

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

# bench-fused: REMOVED — bench_fused_inference fossilized (S-01..S-11 absorbed).
#   See metalForge/fossils/bench/bench_fused_inference.rs for the fossil record.
#   Use bench-ml for current ML benchmarks.

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
