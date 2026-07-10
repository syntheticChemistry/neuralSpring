#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# neuralSpring petalTongue visualization helper.
#
# Usage:
#   ./scripts/visualize.sh                # dump all scenarios to sandbox/scenarios/
#   ./scripts/visualize.sh --live         # start live training dashboard
#   ./scripts/visualize.sh --ecosystem    # start ecosystem dashboard (21 tracks)
#   ./scripts/visualize.sh --compositions # dump + render composition study (5 novel experiments)
#   ./scripts/visualize.sh --render       # dump + launch petalTongue on complete study
#
# Environment:
#   EPOCHS       - epochs for live dashboard (default: 100)
#   INTERVAL_MS  - ms between live pushes    (default: 50)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Post-primordial binary discovery: plasmidBin depot is the sole source.
find_binary() {
    local name="$1"
    local eco="${ECOPRIMALS_ROOT:-$PROJECT_ROOT/../..}"
    local triple machine kernel
    machine=$(uname -m); kernel=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$kernel" in
        linux)  triple="${machine}-unknown-linux-musl" ;;
        darwin) [[ "$machine" = "arm64" ]] && triple="aarch64-apple-darwin" || triple="${machine}-apple-darwin" ;;
        *)      triple="${machine}-unknown-${kernel}" ;;
    esac
    local pb="$eco/infra/plasmidBin/primals/$triple/$name"
    [ -x "$pb" ] && echo "$pb" && return
    local flat="$eco/infra/plasmidBin/primals/$name"
    [ -x "$flat" ] && echo "$flat" && return
    local bin
    bin="$(command -v "$name" 2>/dev/null || true)"
    [ -n "$bin" ] && echo "$bin" && return
    echo >&2 "ERROR: binary '$name' not found in plasmidBin or PATH. Run: membrane plasmid.harvest"
    return 1
}
ECO_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
SCENARIO_DIR="$PROJECT_ROOT/sandbox/scenarios"

find_petaltongue() {
    local triple machine kernel
    machine=$(uname -m); kernel=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$kernel" in
        linux)  triple="${machine}-unknown-linux-musl" ;;
        darwin) [[ "$machine" = "arm64" ]] && triple="aarch64-apple-darwin" || triple="${machine}-apple-darwin" ;;
        *)      triple="${machine}-unknown-${kernel}" ;;
    esac
    local git_plasmid="$ECO_ROOT/infra/plasmidBin/primals"
    for dir in "$git_plasmid/$triple" "$git_plasmid"; do
        [[ -x "$dir/petaltongue" ]] && echo "$dir/petaltongue" && return
    done
    echo ""
}

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
    "$(find_binary dump_neuralspring_scenarios)"
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
            "$(find_binary neuralspring_live_dashboard)"
        ;;

    --ecosystem)
        echo "Building neuralspring_ecosystem_dashboard..."
        cargo build --release --bin neuralspring_ecosystem_dashboard --manifest-path "$PROJECT_ROOT/Cargo.toml"
        echo ""
        "$(find_binary neuralspring_ecosystem_dashboard)"
        ;;

    --compositions)
        dump_scenarios
        COMP="$SCENARIO_DIR/neuralspring-compositions.json"
        PT_BIN="$(find_petaltongue)"
        if [[ -n "$PT_BIN" ]]; then
            echo "Launching petalTongue on composition study..."
            "$PT_BIN" ui --scenario "$COMP"
        else
            echo "petalTongue not found in plasmidBin."
            echo "Render manually: petaltongue ui --scenario $COMP"
        fi
        ;;

    --render)
        dump_scenarios
        COMPLETE="$SCENARIO_DIR/neuralspring-complete-study.json"
        PT_BIN="$(find_petaltongue)"
        if [[ -n "$PT_BIN" ]]; then
            echo "Launching petalTongue..."
            "$PT_BIN" ui --scenario "$COMPLETE"
        else
            echo "petalTongue not found in plasmidBin."
            echo "Render manually: petaltongue ui --scenario $COMPLETE"
        fi
        ;;

    --help|-h)
        head -14 "$0" | tail -12
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [dump|--live|--ecosystem|--compositions|--render|--help]"
        exit 1
        ;;
esac
