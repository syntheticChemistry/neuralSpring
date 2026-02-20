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

# Rust unit tests (42 tests)
test-rust:
    cargo test

# All validation binaries (285 checks)
validate: validate-native validate-barracuda

# neuralSpring-native (43 checks)
validate-native:
    cargo run --bin validate_surrogate
    cargo run --bin validate_transformer
    cargo run --bin validate_metrics

# BarraCUDA primitives (CPU slice + unified Tensor/WGSL path)
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

# Fused pipeline benchmark (CPU + GPU, 4-way comparison)
bench-fused:
    @echo "── Fused Pipeline: CPU ──"
    NEURALSPRING_BACKEND=cpu cargo run --release --bin bench_fused_inference
    @echo ""
    @echo "── Fused Pipeline: GPU ──"
    NEURALSPRING_BACKEND=gpu cargo run --release --bin bench_fused_inference

# Full Python baseline suite (75/75, ~6 min)
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
