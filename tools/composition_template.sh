#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# composition_template.sh — Minimal NUCLEUS composition starter
#
# Copy this file to your spring and fill in the domain hooks.
# The library handles all NUCLEUS wiring; you handle domain logic.
#
# See tools/ttt_composition.sh for a complete reference implementation.
#
# Usage:
#   COMPOSITION_NAME=myspring FAMILY_ID=myspring ./composition_template.sh

set -euo pipefail

# ── 1. Configuration ──────────────────────────────────────────────────
#
# COMPOSITION_NAME is the only required variable. It controls logging
# prefixes, braid session tags, ledger spine naming, and splash display.
#
# REQUIRED_CAPS: capabilities that must be present (discovery fails if missing)
# OPTIONAL_CAPS: capabilities used if available, gracefully degraded if not

COMPOSITION_NAME="${COMPOSITION_NAME:-myspring}"
REQUIRED_CAPS="visualization security"
OPTIONAL_CAPS="compute tensor dag ledger attribution"

# Source the NUCLEUS composition library
# Adjust the path to wherever nucleus_composition_lib.sh lives relative to your script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/nucleus_composition_lib.sh"

# ── 2. Domain State ──────────────────────────────────────────────────
#
# Define your application state here. The lib provides:
#   CURRENT_VERTEX, VERTEX_STACK, STATE_STACK — for DAG tracking
#   SPINE_ID — for ledger spine
#   BRAID_SESSION_TAG, LAST_BRAID_ID — for provenance
#   HOVER_CELL, SENSOR_KEY, SENSOR_CLICK_CELL — from interaction polling

MY_STATE="initial"
RUNNING=true

# ── 3. Hit Testing (override lib default) ─────────────────────────────
#
# The lib calls hit_test_fn(x, y) to map pixel coordinates to logical
# targets (e.g. grid cells, buttons). Return an integer >= 0 for a hit,
# or -1 for no hit. Override this for your layout.

hit_test_fn() {
    local px="$1" py="$2"
    # Example: a single 200x200 area starting at (100,100)
    px="${px%.*}"
    py="${py%.*}"
    if (( px >= 100 && px < 300 && py >= 100 && py < 300 )); then
        echo 0
    else
        echo -1
    fi
}

# ── 4. Domain Hooks ──────────────────────────────────────────────────

domain_init() {
    # Called once at startup after discovery and subscription.
    # Initialize your DAG session, ledger spine, and first render.
    dag_create_session "$COMPOSITION_NAME" "[]"
    ledger_create_spine
    domain_render "Ready — click or press a key"
}

domain_render() {
    local status="${1:-}"
    # Build and push a petalTongue scene graph.
    # Use push_scene "session-id" "$scene_json" from the lib.
    local title
    title=$(make_text_node "title" 230 80 "NUCLEUS $COMPOSITION_NAME" 28 0.95 0.95 1.0)
    local status_node
    status_node=$(make_text_node "status" 230 200 "$status" 18 0.8 0.8 0.85)

    local root
    root=$(printf '"root":{"id":"root","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[],"children":["title","status"],"visible":true,"opacity":1.0,"label":null,"data_source":null}')
    local scene="{\"nodes\":{${root},${title},${status_node}},\"root_id\":\"root\"}"
    push_scene "${COMPOSITION_NAME}-main" "$scene"
}

domain_on_key() {
    local key="$1"
    # Handle keyboard input. Called when SENSOR_KEY is non-empty.
    case "$key" in
        Q|q|Escape)
            log "quit requested"
            RUNNING=false
            ;;
        *)
            log "key pressed: $key"
            # Record to DAG: dag_append_event domain event_name state metadata input_type hover
            dag_append_event "$COMPOSITION_NAME" "keypress" "$MY_STATE" \
                "[{\"key\":\"key\",\"value\":\"$key\"}]" "keyboard" "0"
            # Record braid
            braid_record "keypress" "application/x-${COMPOSITION_NAME}" "$MY_STATE" \
                "{\"key\":\"$key\"}" "keyboard" "0"
            domain_render "Key: $key (DAG depth: ${#VERTEX_STACK[@]})"
            ;;
    esac
}

domain_on_click() {
    local cell="$1"
    # Handle click input. Called when SENSOR_CLICK_CELL >= 0.
    log "clicked target: $cell"
    dag_append_event "$COMPOSITION_NAME" "click" "$MY_STATE" \
        "[{\"key\":\"target\",\"value\":\"$cell\"}]" "click" "$ACCUMULATED_HOVER_MOVES"
    braid_record "click" "application/x-${COMPOSITION_NAME}" "$MY_STATE" \
        "{\"target\":\"$cell\"}" "click" "$ACCUMULATED_HOVER_MOVES"
    ACCUMULATED_HOVER_MOVES=0
    domain_render "Clicked target $cell"
}

domain_on_tick() {
    # Called every poll cycle with no input. Good for animations, 
    # convergence checks, or periodic state updates.
    check_proprioception
}

# ── 5. Main Loop ─────────────────────────────────────────────────────

main() {
    discover_capabilities || { err "Required primals not found"; exit 1; }

    composition_startup "NUCLEUS $COMPOSITION_NAME" "Composition Template"

    subscribe_interactions "click"
    subscribe_sensor_stream

    domain_init

    while $RUNNING; do
        local sensor_batch
        sensor_batch=$(poll_sensor_stream)
        process_sensor_batch "$sensor_batch"

        ACCUMULATED_HOVER_MOVES=$((ACCUMULATED_HOVER_MOVES + SENSOR_HOVER_MOVES))

        if $SENSOR_HOVER_CHANGED; then
            domain_render "Hovering... (target: $HOVER_CELL)"
        fi

        if [[ -n "$SENSOR_KEY" ]]; then
            domain_on_key "$SENSOR_KEY"
        elif [[ "$SENSOR_CLICK_CELL" -ge 0 ]]; then
            domain_on_click "$SENSOR_CLICK_CELL"
        else
            domain_on_tick
            sleep "$POLL_INTERVAL"
        fi
    done

    composition_summary
    composition_teardown "${COMPOSITION_NAME}-main"
}

main
