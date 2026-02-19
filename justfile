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

# Rust unit tests (23 tests)
test-rust:
    cargo test

# hotSpring validation binaries (30 checks)
validate:
    cargo run --bin validate_surrogate
    cargo run --bin validate_transformer
    cargo run --bin validate_metrics

# Full Python baseline suite (75/75, ~6 min)
baselines:
    bash scripts/run_all_baselines.sh

# Auto-fix Python lint issues
fix:
    ruff check --fix control/ scripts/ tests/
    ruff format control/ tests/

# Format both Python and Rust
fmt:
    ruff format control/ tests/
    cargo fmt
