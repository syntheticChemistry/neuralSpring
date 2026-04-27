#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# composition_nucleus.sh — Launch a NUCLEUS composition for any domain
#
# Starts primals from plasmidBin in dependency order with correct socket
# naming for discover_by_capability ({capability}-{FAMILY_ID}.sock).
#
# Usage:
#   ./tools/composition_nucleus.sh start              # full NUCLEUS
#   ./tools/composition_nucleus.sh stop               # graceful shutdown
#   ./tools/composition_nucleus.sh status             # health check
#   COMPOSITION_NAME=ttt ./tools/composition_nucleus.sh start
#
# Configuration (env vars):
#   COMPOSITION_NAME  — identifier (default: "composition")
#   FAMILY_ID         — socket namespace (default: $COMPOSITION_NAME)
#   PRIMAL_LIST       — space-separated primals to start (default: all)
#   PETALTONGUE_LIVE  — "true" to start petalTongue in live GUI mode (default: true)
#   ECOPRIMALS_PLASMID_BIN — path to plasmidBin (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ECO_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"

COMPOSITION_NAME="${COMPOSITION_NAME:-composition}"
FAMILY_ID="${FAMILY_ID:-$COMPOSITION_NAME}"
SOCKET_DIR="${XDG_RUNTIME_DIR:-/tmp}/biomeos"
PID_DIR="/tmp/nucleus-${COMPOSITION_NAME}-pids"
PLASMID_BIN="${ECOPRIMALS_PLASMID_BIN:-$ECO_ROOT/infra/plasmidBin}"
BIN_DIR="$PLASMID_BIN/primals"

PETALTONGUE_LIVE="${PETALTONGUE_LIVE:-true}"
PRIMAL_LIST="${PRIMAL_LIST:-beardog songbird toadstool barracuda rhizocrypt loamspine sweetgrass petaltongue}"

export FAMILY_ID
export BEARDOG_FAMILY_SEED="${BEARDOG_FAMILY_SEED:-$(head -c 32 /dev/urandom | xxd -p | tr -d '\n')}"
export NODE_ID="${NODE_ID:-$(hostname)}"
export BEARDOG_NODE_ID="${BEARDOG_NODE_ID:-$NODE_ID}"

log() { echo "[${COMPOSITION_NAME}-nucleus] $(date +%H:%M:%S) $*"; }
err() { echo "[${COMPOSITION_NAME}-nucleus] ERROR: $*" >&2; }
ok()  { echo "[${COMPOSITION_NAME}-nucleus] OK: $*"; }

wants_primal() {
    local name="$1"
    echo " $PRIMAL_LIST " | grep -q " $name "
}

wait_for_socket() {
    local sock="$1" timeout="${2:-10}" elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        [[ -S "$sock" ]] && return 0
        sleep 0.5
        elapsed=$((elapsed + 1))
    done
    return 1
}

health_check() {
    local sock="$1" method="${2:-health.liveness}"
    echo "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"id\":1}" | \
        timeout 3 socat - "UNIX-CONNECT:$sock" 2>/dev/null
}

save_pid() {
    mkdir -p "$PID_DIR"
    echo "$2" > "$PID_DIR/$1.pid"
}

find_binary() {
    local name="$1"
    if [[ -x "$BIN_DIR/$name" ]]; then
        echo "$BIN_DIR/$name"
        return
    fi
    local release="$ECO_ROOT/primals/$name/target/release/$name"
    [[ -x "$release" ]] && echo "$release" && return
    # CamelCase variant (e.g. petalTongue/target/release/petaltongue)
    for d in "$ECO_ROOT/primals"/*/; do
        local lc
        lc=$(basename "$d" | tr '[:upper:]' '[:lower:]')
        if [[ "$lc" = "$name" ]] && [[ -x "$d/target/release/$name" ]]; then
            echo "$d/target/release/$name"
            return
        fi
    done
    which "$name" 2>/dev/null || true
}

start_primal() {
    local name="$1" binary="$2"; shift 2
    local logfile="/tmp/nucleus-${COMPOSITION_NAME}-${name}.log"
    log "starting $name..."
    setsid "$binary" "$@" > "$logfile" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    save_pid "$name" "$pid"
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        err "$name failed to start. Check $logfile"
        return 1
    fi
    log "$name started (pid=$pid)"
}

sock() { echo "$SOCKET_DIR/$1-${FAMILY_ID}.sock"; }

cmd_start() {
    log "============================================"
    log "  NUCLEUS Composition: $COMPOSITION_NAME"
    log "  family_id:  $FAMILY_ID"
    log "  socket_dir: $SOCKET_DIR"
    log "  bin_dir:    $BIN_DIR"
    log "  primals:    $PRIMAL_LIST"
    log "============================================"
    mkdir -p "$SOCKET_DIR"

    echo "$BEARDOG_FAMILY_SEED" > "$SOCKET_DIR/.family.seed"
    chmod 600 "$SOCKET_DIR/.family.seed"

    # ── Phase 1: Tower Atomic (BearDog + Songbird) ──
    if wants_primal beardog; then
        log "── Phase 1: Tower Atomic ──"
        local beardog_bin
        beardog_bin="$(find_binary beardog)"
        if [[ -n "$beardog_bin" ]]; then
            start_primal beardog "$beardog_bin" server \
                --socket "$(sock beardog)" \
                --family-id "$FAMILY_ID" || { err "beardog required"; return 1; }
            wait_for_socket "$(sock beardog)" 10 || err "beardog socket timeout"
        else
            err "beardog binary not found"; return 1
        fi
    fi

    if wants_primal songbird; then
        local songbird_bin
        songbird_bin="$(find_binary songbird)"
        if [[ -n "$songbird_bin" ]]; then
            SONGBIRD_SECURITY_PROVIDER="$(sock beardog)" \
            SONGBIRD_DISCOVERY_MODE="disabled" \
            BTSP_PROVIDER_SOCKET="$(sock beardog)" \
                start_primal songbird "$songbird_bin" server \
                    --socket "$(sock songbird)" \
                    --beardog-socket "$(sock beardog)" || { err "songbird required"; return 1; }
            wait_for_socket "$(sock songbird)" 10 || err "songbird socket timeout"
        else
            err "songbird binary not found"; return 1
        fi
    fi

    # ── Phase 2: Compute ──
    if wants_primal toadstool || wants_primal barracuda; then
        log "── Phase 2: Compute Services ──"
    fi

    if wants_primal toadstool; then
        local toadstool_bin
        toadstool_bin="$(find_binary toadstool)"
        if [[ -n "$toadstool_bin" ]]; then
            TOADSTOOL_SOCKET="$(sock toadstool)" \
            TOADSTOOL_FAMILY_ID="$FAMILY_ID" \
            TOADSTOOL_SECURITY_WARNING_ACKNOWLEDGED="1" \
            NESTGATE_SOCKET="$(sock nestgate)" \
                start_primal toadstool "$toadstool_bin" server || log "WARN: toadstool failed"
            wait_for_socket "$(sock toadstool)" 8 || log "WARN: toadstool socket not ready"
        else
            log "WARN: toadstool binary not found"
        fi
    fi

    if wants_primal barracuda; then
        local barracuda_bin
        barracuda_bin="$(find_binary barracuda)"
        if [[ -n "$barracuda_bin" ]]; then
            BARRACUDA_FAMILY_ID="$FAMILY_ID" \
            BEARDOG_SOCKET="$(sock beardog)" \
            SONGBIRD_SOCKET="$(sock songbird)" \
                start_primal barracuda "$barracuda_bin" server || log "WARN: barracuda failed"
            wait_for_socket "$SOCKET_DIR/barracuda-${FAMILY_ID}.sock" 8 || \
                wait_for_socket "$SOCKET_DIR/math-${FAMILY_ID}.sock" 5 || \
                log "WARN: barracuda socket not ready"
            if [[ ! -e "$SOCKET_DIR/barracuda-${FAMILY_ID}.sock" && -S "$SOCKET_DIR/math-${FAMILY_ID}.sock" ]]; then
                ln -sf "math-${FAMILY_ID}.sock" "$SOCKET_DIR/barracuda-${FAMILY_ID}.sock" 2>/dev/null || true
            fi
        else
            log "WARN: barracuda binary not found"
        fi
    fi

    # ── Phase 3: Provenance Trio ──
    if wants_primal rhizocrypt || wants_primal loamspine || wants_primal sweetgrass; then
        log "── Phase 3: Provenance Trio ──"
    fi

    if wants_primal rhizocrypt; then
        local rhizocrypt_bin
        rhizocrypt_bin="$(find_binary rhizocrypt)"
        if [[ -n "$rhizocrypt_bin" ]]; then
            RHIZOCRYPT_SOCKET="$(sock rhizocrypt)" \
            BIOMEOS_SOCKET_DIR="$SOCKET_DIR" \
            BEARDOG_SOCKET="$(sock beardog)" \
            BTSP_PROVIDER_SOCKET="$(sock beardog)" \
                start_primal rhizocrypt "$rhizocrypt_bin" server || log "WARN: rhizocrypt failed"
            wait_for_socket "$(sock rhizocrypt)" 12 || \
                wait_for_socket "$SOCKET_DIR/rhizocrypt.sock" 4 || \
                log "WARN: rhizocrypt socket not ready"
        else
            log "WARN: rhizocrypt binary not found"
        fi
    fi

    if wants_primal loamspine; then
        local loamspine_bin
        loamspine_bin="$(find_binary loamspine)"
        if [[ -n "$loamspine_bin" ]]; then
            LOAMSPINE_SOCKET="$(sock loamspine)" \
            BIOMEOS_SOCKET_DIR="$SOCKET_DIR" \
            BEARDOG_SOCKET="$(sock beardog)" \
            RHIZOCRYPT_SOCKET="$(sock rhizocrypt)" \
            BTSP_PROVIDER_SOCKET="$(sock beardog)" \
            BIOMEOS_FAMILY_ID="$FAMILY_ID" \
                start_primal loamspine "$loamspine_bin" server || log "WARN: loamspine failed"
            wait_for_socket "$(sock loamspine)" 8 || log "WARN: loamspine socket not ready"
        else
            log "WARN: loamspine binary not found"
        fi
    fi

    if wants_primal sweetgrass; then
        local sweetgrass_bin
        sweetgrass_bin="$(find_binary sweetgrass)"
        if [[ -n "$sweetgrass_bin" ]]; then
            SWEETGRASS_SOCKET="$(sock sweetgrass)" \
            BIOMEOS_SOCKET_DIR="$SOCKET_DIR" \
            BEARDOG_SOCKET="$(sock beardog)" \
            BTSP_PROVIDER_SOCKET="$(sock beardog)" \
                start_primal sweetgrass "$sweetgrass_bin" server || log "WARN: sweetgrass failed"
            wait_for_socket "$(sock sweetgrass)" 8 || log "WARN: sweetgrass socket not ready"
        else
            log "WARN: sweetgrass binary not found"
        fi
    fi

    # ── Phase 4: petalTongue ──
    if wants_primal petaltongue; then
        log "── Phase 4: petalTongue ──"
        local petaltongue_bin
        petaltongue_bin="$ECO_ROOT/primals/petalTongue/target/release/petaltongue"
        [[ -x "$petaltongue_bin" ]] || petaltongue_bin="$(find_binary petaltongue)"

        if [[ -x "$petaltongue_bin" ]]; then
            local pt_logfile="/tmp/nucleus-${COMPOSITION_NAME}-petaltongue.log"
            local pt_mode="server"
            [[ "$PETALTONGUE_LIVE" = "true" ]] && pt_mode="live"

            if [[ "$pt_mode" = "live" ]]; then
                log "starting petaltongue (live, no setsid)..."
                DISPLAY="${DISPLAY:-:1}" \
                PETALTONGUE_SOCKET="$(sock petaltongue)" \
                FAMILY_ID="$FAMILY_ID" \
                BEARDOG_FAMILY_SEED="$BEARDOG_FAMILY_SEED" \
                AWAKENING_ENABLED=false \
                    "$petaltongue_bin" live --socket "$(sock petaltongue)" > "$pt_logfile" 2>&1 &
                local pt_pid=$!
                save_pid petaltongue "$pt_pid"
                sleep 2
            else
                PETALTONGUE_SOCKET="$(sock petaltongue)" \
                FAMILY_ID="$FAMILY_ID" \
                BEARDOG_FAMILY_SEED="$BEARDOG_FAMILY_SEED" \
                AWAKENING_ENABLED=false \
                    start_primal petaltongue "$petaltongue_bin" server \
                        --socket "$(sock petaltongue)" || { err "petaltongue failed"; return 1; }
            fi

            if [[ "$pt_mode" = "live" ]]; then
                if ! kill -0 "$pt_pid" 2>/dev/null; then
                    err "petaltongue failed to start. Check $pt_logfile"
                    return 1
                fi
                log "petaltongue started (pid=$pt_pid)"
            fi
            wait_for_socket "$(sock petaltongue)" 10 || err "petaltongue socket timeout"
        else
            err "petaltongue binary not found"; return 1
        fi
    fi

    # ── Capability domain symlinks ──
    log "── Creating capability aliases ──"
    local -A domain_map=(
        [security]="beardog-${FAMILY_ID}.sock"
        [crypto]="beardog-${FAMILY_ID}.sock"
        [btsp]="beardog-${FAMILY_ID}.sock"
        [discovery]="songbird-${FAMILY_ID}.sock"
        [compute]="toadstool-${FAMILY_ID}.sock"
        [tensor]="barracuda-${FAMILY_ID}.sock"
        [math]="barracuda-${FAMILY_ID}.sock"
        [provenance]="rhizocrypt-${FAMILY_ID}.sock"
        [dag]="rhizocrypt-${FAMILY_ID}.sock"
        [ledger]="loamspine-${FAMILY_ID}.sock"
        [attribution]="sweetgrass-${FAMILY_ID}.sock"
        [visualization]="petaltongue-${FAMILY_ID}.sock"
    )
    for domain in "${!domain_map[@]}"; do
        local target="${domain_map[$domain]}"
        local alias_path="$SOCKET_DIR/${domain}-${FAMILY_ID}.sock"
        if [[ -S "$SOCKET_DIR/$target" ]] && [[ ! -e "$alias_path" ]]; then
            ln -sf "$target" "$alias_path" 2>/dev/null && \
                log "  ${domain}-${FAMILY_ID}.sock -> $target" || true
        fi
    done

    # ── Health summary ──
    log "── NUCLEUS Health Check ──"
    local healthy=0 total=0
    for primal in $PRIMAL_LIST; do
        total=$((total + 1))
        local s="$(sock "$primal")"
        if [[ -S "$s" ]]; then
            local resp
            resp=$(health_check "$s" 2>/dev/null || true)
            if echo "$resp" | grep -q '"alive"\|"ok"\|"healthy"'; then
                ok "$primal: healthy"
                healthy=$((healthy + 1))
            elif [[ -n "$resp" ]]; then
                log "$primal: responding (non-standard) — $(echo "$resp" | head -c 80)"
                healthy=$((healthy + 1))
            else
                log "WARN: $primal socket exists but no health response"
            fi
        else
            log "WARN: $primal socket not found at $s"
        fi
    done
    log "── Result: $healthy/$total primals healthy ──"
    ok "NUCLEUS ready. Run your composition script."
}

cmd_stop() {
    log "Stopping NUCLEUS $COMPOSITION_NAME..."
    local stop_order=""
    for name in petaltongue sweetgrass loamspine rhizocrypt barracuda toadstool songbird beardog; do
        wants_primal "$name" && stop_order="$stop_order $name"
    done
    for name in $stop_order; do
        local pidfile="$PID_DIR/$name.pid"
        if [[ -f "$pidfile" ]]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && log "stopped $name (pid=$pid)" || true
            fi
            rm -f "$pidfile"
        fi
    done
    rm -f "$SOCKET_DIR"/*-${FAMILY_ID}.sock 2>/dev/null || true
    ok "NUCLEUS stopped"
}

cmd_status() {
    log "── NUCLEUS Status (family=$FAMILY_ID) ──"
    for primal in $PRIMAL_LIST; do
        local s="$(sock "$primal")"
        if [[ -S "$s" ]]; then
            local resp
            resp=$(health_check "$s" 2>/dev/null || true)
            if [[ -n "$resp" ]]; then
                ok "$primal: $(echo "$resp" | head -c 120)"
            else
                log "$primal: socket exists, no response"
            fi
        else
            log "$primal: not running"
        fi
    done
}

case "${1:-start}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    restart) cmd_stop; sleep 2; cmd_start ;;
    *) err "Usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
