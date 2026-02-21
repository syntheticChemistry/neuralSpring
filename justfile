# neuralSpring task runner
# Install: cargo install just  (or: sudo apt install just)
# Usage:   just check           — run all quality gates
#          just test            — Python + Rust tests only
#          just lint            — lint + format check only
#          just baselines       — full Python baseline suite (~6 min)

# Run all quality gates (fast — excludes baselines)
check: lint test validate
    @echo ""
    @echo "━━━ ALL GATES PASS ━━━"

# Python + Rust tests
test: test-python test-rust

# Lint and format checks
lint: lint-python lint-rust

# Python lint + format
lint-python:
    ruff check control/ scripts/ tests/
    ruff format --check control/ tests/

# Rust lint + format + doc
lint-rust:
    cargo clippy -- -D warnings
    cargo fmt --check
    cargo doc --no-deps

# Python unit tests (48 tests)
test-python:
    python3 -m pytest tests/ -v --tb=short

# Rust unit tests (181 unit + 6 doc-tests, 90.55% coverage)
test-rust:
    cargo test

# All validation binaries (67 via validate_all)
validate: validate-native validate-native-papers validate-barracuda validate-barracuda-cpu

# neuralSpring quick (3 bins: surrogate, transformer, metrics)
validate-native:
    cargo run --bin validate_surrogate
    cargo run --bin validate_transformer
    cargo run --bin validate_metrics

# neuralSpring paper validators (18 bins)
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

# Run everything (or: cargo run --release --bin validate_all)
validate-all: validate-native validate-native-papers validate-barracuda validate-barracuda-cpu
    @echo "━━━ All validation binaries PASS ━━━"

# BarraCUDA primitives (10 bins, 242 checks)
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

# Phase 2: BarraCUDA CPU ports (17 bins)
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

# ML inference validation only (MLP + Transformer)
validate-ml:
    cargo run --bin validate_barracuda_ml_inference

# Tensor on explicit CPU software backend
validate-tensor-cpu:
    NEURALSPRING_BACKEND=cpu cargo run --bin validate_barracuda_tensor

# Tensor on explicit GPU backend
validate-tensor-gpu:
    NEURALSPRING_BACKEND=gpu cargo run --bin validate_barracuda_tensor

# Tensor on BOTH — proves WGSL math is universal across hardware
validate-tensor-all:
    @echo "── Tensor validation: CPU software backend ──"
    NEURALSPRING_BACKEND=cpu cargo run --bin validate_barracuda_tensor
    @echo ""
    @echo "── Tensor validation: GPU backend ──"
    NEURALSPRING_BACKEND=gpu cargo run --bin validate_barracuda_tensor
    @echo ""
    @echo "── Tensor WGSL math is universal across hardware ──"

# Benchmark tensor ops on current backend
bench-tensor:
    cargo run --release --bin bench_barracuda_tensor

# Benchmark CPU vs GPU side by side
bench-tensor-compare:
    @echo "── Benchmark: CPU software backend ──"
    NEURALSPRING_BACKEND=cpu cargo run --release --bin bench_barracuda_tensor
    @echo ""
    @echo "── Benchmark: GPU backend ──"
    NEURALSPRING_BACKEND=gpu cargo run --release --bin bench_barracuda_tensor

# Benchmark ML inference (MLP + Transformer)
bench-ml:
    cargo run --release --bin bench_mlp_inference
    cargo run --release --bin bench_transformer_block

# bench-fused: REMOVED — bench_fused_inference fossilized (S-01..S-11 absorbed).
#   See metalForge/fossils/bench/bench_fused_inference.rs for the fossil record.
#   Use bench-ml for current ML benchmarks.

# Full Python baseline suite (206/206, ~10 min)
baselines:
    bash scripts/run_all_baselines.sh

# Library test coverage report (requires cargo-llvm-cov)
coverage:
    cargo llvm-cov --lib --html
    @echo "Coverage report: target/llvm-cov/html/index.html"

# Coverage as JSON (for CI thresholds)
coverage-json:
    cargo llvm-cov --lib --json --output-path target/llvm-cov/coverage.json

# Auto-fix Python lint issues
fix:
    ruff check --fix control/ scripts/ tests/
    ruff format control/ tests/

# Format both Python and Rust
fmt:
    ruff format control/ tests/
    cargo fmt
