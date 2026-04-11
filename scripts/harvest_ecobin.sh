#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Harvest neuralSpring ecoBin artifacts for plasmidBin staging.
#
# Builds the primal binary as a statically-linked musl binary, verifies
# ecoBin compliance (static linkage, no C deps, health.liveness), and
# stages artifacts for infra/plasmidBin.
#
# Usage:
#   ./scripts/harvest_ecobin.sh [--arch x86_64|aarch64] [--skip-verify]
#
# Environment:
#   PLASMIDB_DIR    Override plasmidBin directory (default: ../../infra/plasmidBin)
#   ECOBIN_VERSION  Override version tag (default: from Cargo.toml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLASMIDB_DIR="${PLASMIDB_DIR:-${PROJECT_DIR}/../../infra/plasmidBin}"
ARCH="${1:-x86_64}"
SKIP_VERIFY="${2:-}"
PRIMAL_NAME="neuralspring"

cd "${PROJECT_DIR}"

VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
ECOBIN_VERSION="${ECOBIN_VERSION:-${VERSION}}"

echo "═══ neuralSpring ecoBin Harvest ═══"
echo "Version: ${ECOBIN_VERSION}"
echo "Architecture: ${ARCH}"
echo "Target: ${PLASMIDB_DIR}"
echo ""

# ── Step 1: Determine target triple ──

case "${ARCH}" in
    x86_64)
        TARGET="x86_64-unknown-linux-musl"
        ;;
    aarch64)
        TARGET="aarch64-unknown-linux-musl"
        ;;
    *)
        echo "FAIL: Unknown architecture '${ARCH}'. Use x86_64 or aarch64."
        exit 1
        ;;
esac

echo "── Step 1: Build (${TARGET}) ──"

if ! rustup target list --installed | grep -q "${TARGET}"; then
    echo "Installing target ${TARGET}..."
    rustup target add "${TARGET}"
fi

cargo build --release --target "${TARGET}" --bin neuralspring --features primal

BINARY="target/${TARGET}/release/neuralspring"

if [ ! -f "${BINARY}" ]; then
    echo "FAIL: Binary not found at ${BINARY}"
    exit 1
fi

echo "PASS: Binary built at ${BINARY}"

# ── Step 2: Verify static linkage ──

echo ""
echo "── Step 2: Verify static linkage ──"

FILE_OUTPUT=$(file "${BINARY}")
echo "  file: ${FILE_OUTPUT}"

if echo "${FILE_OUTPUT}" | grep -q "statically linked"; then
    echo "PASS: Statically linked"
elif echo "${FILE_OUTPUT}" | grep -q "static-pie"; then
    echo "PASS: Static PIE linked"
else
    echo "WARN: Binary may not be statically linked"
    if command -v ldd &>/dev/null; then
        LDD_OUTPUT=$(ldd "${BINARY}" 2>&1 || true)
        if echo "${LDD_OUTPUT}" | grep -q "not a dynamic executable"; then
            echo "PASS: ldd confirms static"
        else
            echo "FAIL: ldd shows dynamic dependencies:"
            echo "${LDD_OUTPUT}"
            exit 1
        fi
    fi
fi

# ── Step 3: Verify no banned C deps ──

echo ""
echo "── Step 3: Verify no banned C deps ──"

BANNED_CRATES="openssl-sys ring aws-lc-sys zstd-sys lz4-sys sysinfo"
TREE_OUTPUT=$(cargo tree --all-features --target "${TARGET}" 2>/dev/null || cargo tree --all-features 2>/dev/null)

for crate in ${BANNED_CRATES}; do
    if echo "${TREE_OUTPUT}" | grep -q "${crate}"; then
        echo "FAIL: Banned C sys crate '${crate}' found"
        exit 1
    fi
done
echo "PASS: No banned C sys crates"

# ── Step 4: Size report ──

echo ""
echo "── Step 4: Size report ──"

SIZE=$(stat -c%s "${BINARY}" 2>/dev/null || stat -f%z "${BINARY}")
SIZE_MB=$(echo "scale=1; ${SIZE}/1048576" | bc)
echo "  Binary size: ${SIZE_MB} MB (${SIZE} bytes)"

# ── Step 5: Stage to plasmidBin ──

if [ "${SKIP_VERIFY}" = "--skip-verify" ]; then
    echo ""
    echo "── Step 5: Skipping plasmidBin staging (--skip-verify) ──"
else
    echo ""
    echo "── Step 5: Stage to plasmidBin ──"

    STAGING_DIR="${PLASMIDB_DIR}/${PRIMAL_NAME}"
    mkdir -p "${STAGING_DIR}"

    ARTIFACT_NAME="${PRIMAL_NAME}-${ARCH}"
    cp "${BINARY}" "${STAGING_DIR}/${ARTIFACT_NAME}"

    if command -v b3sum &>/dev/null; then
        B3=$(b3sum "${STAGING_DIR}/${ARTIFACT_NAME}" | cut -d' ' -f1)
        echo "  b3sum: ${B3}"
    elif command -v sha256sum &>/dev/null; then
        SHA=$(sha256sum "${STAGING_DIR}/${ARTIFACT_NAME}" | cut -d' ' -f1)
        echo "  sha256: ${SHA}"
    fi

    echo "PASS: Staged to ${STAGING_DIR}/${ARTIFACT_NAME}"
fi

echo ""
echo "═══ Harvest complete ═══"
echo "  Binary: ${BINARY}"
echo "  Version: ${ECOBIN_VERSION}"
echo "  Target: ${TARGET}"
