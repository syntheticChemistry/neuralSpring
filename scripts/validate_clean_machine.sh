#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Clean-machine NUCLEUS validation (Level 6).
#
# Validates that neuralSpring's primal proof works against a NUCLEUS
# deployed entirely from plasmidBin ecobins — no source tree, no path
# deps, no barraCuda crate on disk.
#
# Prerequisites:
#   1. plasmidBin ecobins deployed (all 9 primals as static binaries)
#   2. biomeOS orchestrator running
#   3. neuralspring binary built and on PATH (or provide via NEURALSPRING_BIN)
#
# Usage:
#   ./scripts/validate_clean_machine.sh [--tier 2|3|all]
#
# Environment:
#   BIOMEOS_ORCHESTRATOR_SOCKET   Override orchestrator socket path
#   NEURALSPRING_BIN              Path to neuralspring binary (default: neuralspring)
#   NUCLEUS_GRAPH                 Path to proto-nucleate graph TOML
#   TIER                          Validation tier (2=Rust proof, 3=IPC, all=both)
#
# Exit codes:  0 = pass, 1 = fail, 2 = skip (primals not available)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NEURALSPRING_BIN="${NEURALSPRING_BIN:-neuralspring}"
TIER="${TIER:-all}"
PASS=0
FAIL=0
SKIP=0

if [ "${1:-}" = "--tier" ]; then
    TIER="${2:-all}"
fi

echo "═══ neuralSpring Clean-Machine NUCLEUS Validation ═══"
echo "Tier: ${TIER}"
echo "Binary: ${NEURALSPRING_BIN}"
echo ""

# ── Step 1: Verify neuralspring binary exists ──

echo "── Step 1: Binary check ──"

if ! command -v "${NEURALSPRING_BIN}" &>/dev/null && [ ! -f "${NEURALSPRING_BIN}" ]; then
    echo "FAIL: neuralspring binary not found: ${NEURALSPRING_BIN}"
    echo "  Set NEURALSPRING_BIN or add neuralspring to PATH"
    exit 1
fi
echo "PASS: Binary found"

# ── Step 2: Verify primal health ──

echo ""
echo "── Step 2: Primal health check ──"

"${NEURALSPRING_BIN}" health 2>/dev/null && echo "PASS: health" || {
    echo "FAIL: neuralspring health check failed"
    exit 1
}

# ── Step 3: Check biomeOS orchestrator ──

echo ""
echo "── Step 3: biomeOS orchestrator ──"

ORCH_SOCK="${BIOMEOS_ORCHESTRATOR_SOCKET:-}"
if [ -z "${ORCH_SOCK}" ]; then
    if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
        ORCH_SOCK="${XDG_RUNTIME_DIR}/biomeos/biomeos.sock"
    else
        ORCH_SOCK="/tmp/biomeos/biomeos.sock"
    fi
fi

if [ -S "${ORCH_SOCK}" ]; then
    echo "PASS: Orchestrator socket found at ${ORCH_SOCK}"
else
    echo "SKIP: Orchestrator not found at ${ORCH_SOCK}"
    echo "  Set BIOMEOS_ORCHESTRATOR_SOCKET or start biomeOS"
    SKIP=$((SKIP + 1))
fi

# ── Step 4: Run validation tiers ──

echo ""
echo "── Step 4: Validation tiers ──"

run_validator() {
    local name="$1"
    local bin_name="$2"
    local binary="${PROJECT_DIR}/target/release/${bin_name}"

    if [ ! -f "${binary}" ]; then
        binary="${PROJECT_DIR}/target/debug/${bin_name}"
    fi
    if [ ! -f "${binary}" ]; then
        echo "  SKIP ${name}: binary not found (${bin_name})"
        SKIP=$((SKIP + 1))
        return
    fi

    echo "  Running: ${name}..."
    if "${binary}" 2>&1; then
        echo "  PASS: ${name}"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        if [ "${exit_code}" -eq 2 ]; then
            echo "  SKIP: ${name} (honest skip — primals not running)"
            SKIP=$((SKIP + 1))
        else
            echo "  FAIL: ${name} (exit ${exit_code})"
            FAIL=$((FAIL + 1))
        fi
    fi
}

if [ "${TIER}" = "2" ] || [ "${TIER}" = "all" ]; then
    echo ""
    echo "── Tier 2: Rust proof (library validation) ──"
    run_validator "nucleus_composition" "validate_nucleus_composition"
    run_validator "composition_evolution" "validate_composition_evolution"
fi

if [ "${TIER}" = "3" ] || [ "${TIER}" = "all" ]; then
    echo ""
    echo "── Tier 3: IPC proof (primal composition) ──"
    run_validator "science_composition" "validate_science_composition"
    run_validator "proto_nucleate_capabilities" "validate_proto_nucleate_capabilities"
fi

# ── Step 5: Summary ──

echo ""
echo "═══ Summary ═══"
echo "  Passed:  ${PASS}"
echo "  Failed:  ${FAIL}"
echo "  Skipped: ${SKIP}"

if [ "${FAIL}" -gt 0 ]; then
    echo ""
    echo "FAIL: ${FAIL} validation(s) failed"
    exit 1
elif [ "${PASS}" -gt 0 ]; then
    echo ""
    echo "PASS: all exercised validations passed"
    exit 0
else
    echo ""
    echo "SKIP: no validations could be exercised"
    exit 2
fi
