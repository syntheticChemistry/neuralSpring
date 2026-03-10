#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# neuralSpring petalTongue visualization helper.
#
# Usage:
#   ./scripts/visualize.sh              # dump all scenarios to sandbox/scenarios/
#   ./scripts/visualize.sh --live       # start live training dashboard
#   ./scripts/visualize.sh --ecosystem  # start ecosystem dashboard (16 tracks)
#   ./scripts/visualize.sh --render     # dump + launch petalTongue on complete study
#
# Environment:
#   EPOCHS       - epochs for live dashboard (default: 100)
#   INTERVAL_MS  - ms between live pushes    (default: 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENARIO_DIR="$PROJECT_ROOT/sandbox/scenarios"

MODE="${1:-dump}"

build_dump() {
    echo "Building dump_neuralspring_scenarios..."
    cargo build --release --bin dump_neuralspring_scenarios --manifest-path "$PROJECT_ROOT/Cargo.toml"
}

build_live() {
    echo "Building neuralspring_live_dashboard..."
    cargo build --release --bin neuralspring_live_dashboard --manifest-path "$PROJECT_ROOT/Cargo.toml"
}

dump_scenarios() {
    build_dump
    echo ""
    "$PROJECT_ROOT/target/release/dump_neuralspring_scenarios"
    echo ""
    echo "Scenarios ready in $SCENARIO_DIR/"
}

case "$MODE" in
    dump|--dump)
        dump_scenarios
        ;;

    --live)
        build_live
        echo ""
        EPOCHS="${EPOCHS:-100}" INTERVAL_MS="${INTERVAL_MS:-50}" \
            "$PROJECT_ROOT/target/release/neuralspring_live_dashboard"
        ;;

    --ecosystem)
        echo "Building neuralspring_ecosystem_dashboard..."
        cargo build --release --bin neuralspring_ecosystem_dashboard --manifest-path "$PROJECT_ROOT/Cargo.toml"
        echo ""
        "$PROJECT_ROOT/target/release/neuralspring_ecosystem_dashboard"
        ;;

    --render)
        dump_scenarios
        COMPLETE="$SCENARIO_DIR/neuralspring-complete-study.json"
        if command -v petaltongue &>/dev/null; then
            echo "Launching petalTongue..."
            petaltongue ui --scenario "$COMPLETE"
        else
            echo "petalTongue not found in PATH."
            echo "Render manually: petaltongue ui --scenario $COMPLETE"
        fi
        ;;

    --help|-h)
        head -14 "$0" | tail -12
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [dump|--live|--ecosystem|--render|--help]"
        exit 1
        ;;
esac
