#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Download Militzer FPEOS tables (Berkeley, open C++/Python).
# Source: https://militzer.berkeley.edu/FPEOS/
# Citation: Militzer et al., PRE 103, 013203 (2021)
#
# Data format per line:
#   f=<formula> N=<atoms> rho[g/cc]=<density> V[A^3]=<volume>
#   T[K]=<temp> P[GPa]=<pressure> <P_err> E[Ha]=<energy> <E_err>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/fpeos_data"
BASE_URL="https://militzer.berkeley.edu/FPEOS/files"

mkdir -p "$DATA_DIR"

echo "=== FPEOS Table Acquisition for nW-02 ==="
echo "Source: Militzer et al., PRE 103, 013203 (2021)"
echo "URL: https://militzer.berkeley.edu/FPEOS/"
echo ""

declare -A FILES=(
    [H]="H_EOS_09-18-20.txt"
    [He]="He_EOS_09-18-20.txt"
    [C]="C_EOS_09-18-20.txt"
)

for element in H He C; do
    fname="${FILES[$element]}"
    echo "Downloading ${element} EOS table (${fname})..."
    if curl -fsSL "${BASE_URL}/${fname}" -o "${DATA_DIR}/${fname}"; then
        lines=$(wc -l < "${DATA_DIR}/${fname}")
        echo "  ${element}: OK (${lines} lines)"
    else
        echo "  WARNING: Could not download ${element}."
        echo "  Manual download: ${BASE_URL}/${fname}"
    fi
done

echo ""
echo "=== Download complete ==="
ls -la "${DATA_DIR}/" 2>/dev/null || true
