#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# neural_composition.sh — Agent-Driven Composition + AI Feedback Loops
#
# neuralSpring's assigned exploration lane (Phase 46): Squirrel-mediated
# interactions through the full NUCLEUS stack. This script explores:
#
#   1. Squirrel-mediated composition — inference.complete / inference.embed
#      via IPC through the primal stack
#   2. Inference pipeline — how embedding + completion fit into DAG state
#   3. Model decision audit — braid provenance for AI decisions
#   4. Agent feedback loops — act → observe → adjust via proprioception
#
# Prerequisites:
#   - NUCLEUS running (./tools/composition_nucleus.sh start)
#   - Squirrel primal available (add to PRIMAL_LIST if not default)
#
# Usage:
#   COMPOSITION_NAME=neuralspring ./tools/neural_composition.sh
#   FAMILY_ID=neuralspring-agent ./tools/neural_composition.sh

set -euo pipefail

# ── 1. Configuration ──────────────────────────────────────────────────

COMPOSITION_NAME="${COMPOSITION_NAME:-neuralspring}"
REQUIRED_CAPS="visualization security ai"
OPTIONAL_CAPS="compute tensor dag ledger attribution inference"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/nucleus_composition_lib.sh"

# ── 2. Agent State ────────────────────────────────────────────────────

AGENT_STATE="idle"
INFERENCE_COUNT=0
EMBED_COUNT=0
REASON_COUNT=0
LAST_INFERENCE_RESULT=""
LAST_EMBED_RESULT=""
LAST_PROMPT=""
RUNNING=true
TICK_COUNT=0
AUTO_REASON_INTERVAL=20

# ── 3. Hit Testing ───────────────────────────────────────────────────

hit_test_fn() {
    local px="$1" py="$2"
    px="${px%.*}"
    py="${py%.*}"
    # Dashboard layout: 3 action buttons across the top
    # [Infer: 50-200] [Embed: 220-370] [Reason: 390-540]
    if (( py >= 60 && py < 110 )); then
        if (( px >= 50 && px < 200 )); then echo 0; return; fi   # Infer
        if (( px >= 220 && px < 370 )); then echo 1; return; fi  # Embed
        if (( px >= 390 && px < 540 )); then echo 2; return; fi  # Reason
    fi
    echo -1
}

# ── 4. Squirrel IPC Helpers ──────────────────────────────────────────

agent_inference_complete() {
    local prompt="$1"
    local max_tokens="${2:-64}"
    local socket
    socket=$(cap_socket "ai") || socket=$(cap_socket "inference") || true
    if [[ -z "$socket" ]]; then
        warn "Squirrel not available — cannot run inference"
        LAST_INFERENCE_RESULT="[squirrel_unavailable]"
        return 1
    fi
    local params
    params=$(printf '{"prompt":"%s","model":"default","max_tokens":%d}' \
        "$prompt" "$max_tokens")
    local result
    result=$(send_rpc_quiet "$socket" "inference.complete" "$params") || true
    if [[ -n "$result" && "$result" != "null" ]]; then
        LAST_INFERENCE_RESULT="$result"
        INFERENCE_COUNT=$((INFERENCE_COUNT + 1))
        return 0
    else
        LAST_INFERENCE_RESULT="[no_response]"
        return 1
    fi
}

agent_inference_embed() {
    local text="$1"
    local socket
    socket=$(cap_socket "ai") || socket=$(cap_socket "inference") || true
    if [[ -z "$socket" ]]; then
        warn "Squirrel not available — cannot embed"
        LAST_EMBED_RESULT="[squirrel_unavailable]"
        return 1
    fi
    local params
    params=$(printf '{"text":"%s","model":"default"}' "$text")
    local result
    result=$(send_rpc_quiet "$socket" "inference.embed" "$params") || true
    if [[ -n "$result" && "$result" != "null" ]]; then
        LAST_EMBED_RESULT="$result"
        EMBED_COUNT=$((EMBED_COUNT + 1))
        return 0
    else
        LAST_EMBED_RESULT="[no_response]"
        return 1
    fi
}

# ── 5. DAG + Braid Provenance Helpers ────────────────────────────────

record_agent_action() {
    local action="$1" detail="$2" input_type="${3:-agent}" hover="${4:-0}"

    dag_append_event "$COMPOSITION_NAME" "$action" "$AGENT_STATE" \
        "[{\"key\":\"action\",\"value\":\"$action\"},{\"key\":\"detail\",\"value\":\"$detail\"},{\"key\":\"inference_count\",\"value\":\"$INFERENCE_COUNT\"},{\"key\":\"embed_count\",\"value\":\"$EMBED_COUNT\"}]" \
        "$input_type" "$hover"

    braid_record "$action" "application/x-neuralspring-agent" "$AGENT_STATE" \
        "{\"action\":\"$action\",\"detail\":\"$detail\",\"tick\":$TICK_COUNT}" \
        "$input_type" "$hover"
}

# ── 6. Domain Hooks ──────────────────────────────────────────────────

domain_init() {
    dag_create_session "$COMPOSITION_NAME" \
        "[{\"key\":\"lane\",\"value\":\"agent-driven-composition\"},{\"key\":\"squirrel\",\"value\":\"required\"}]"
    ledger_create_spine

    AGENT_STATE="initializing"
    log "probing Squirrel inference pipeline..."

    if agent_inference_complete "Hello from neuralSpring agent. Respond with OK." 16; then
        AGENT_STATE="ready"
        ok "Squirrel inference pipeline live"
        record_agent_action "init" "squirrel_live" "system"
    else
        AGENT_STATE="degraded"
        warn "Squirrel unavailable — inference will be simulated"
        record_agent_action "init" "squirrel_unavailable" "system"
    fi

    domain_render "Agent $AGENT_STATE | I=infer E=embed R=reason Q=quit"
}

domain_render() {
    local status="${1:-}"
    local title
    title=$(make_text_node "title" 300 30 "neuralSpring Agent Composition" 24 0.9 0.95 1.0)

    local btn_infer
    btn_infer=$(make_text_node "btn_infer" 125 80 "[I] Infer ($INFERENCE_COUNT)" 16 0.7 1.0 0.7)
    local btn_embed
    btn_embed=$(make_text_node "btn_embed" 295 80 "[E] Embed ($EMBED_COUNT)" 16 0.7 0.7 1.0)
    local btn_reason
    btn_reason=$(make_text_node "btn_reason" 465 80 "[R] Reason ($REASON_COUNT)" 16 1.0 0.7 0.7)

    local state_display="State: $AGENT_STATE | Tick: $TICK_COUNT"
    local state_node
    state_node=$(make_text_node "state" 300 130 "$state_display" 14 0.8 0.8 0.85)

    local status_node
    status_node=$(make_text_node "status" 300 160 "$status" 14 0.75 0.75 0.8)

    local dag_info="DAG depth: ${#VERTEX_STACK[@]} | Braids: $((INFERENCE_COUNT + EMBED_COUNT + REASON_COUNT))"
    local dag_node
    dag_node=$(make_text_node "dag_info" 300 190 "$dag_info" 12 0.6 0.65 0.7)

    local last_result="${LAST_INFERENCE_RESULT:0:80}"
    [[ ${#LAST_INFERENCE_RESULT} -gt 80 ]] && last_result="${last_result}..."
    local result_node
    result_node=$(make_text_node "result" 300 230 "Last: $last_result" 12 0.65 0.7 0.65)

    local root
    root=$(printf '"root":{"id":"root","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[],"children":["title","btn_infer","btn_embed","btn_reason","state","status","dag_info","result"],"visible":true,"opacity":1.0,"label":null,"data_source":null}')
    local scene="{\"nodes\":{${root},${title},${btn_infer},${btn_embed},${btn_reason},${state_node},${status_node},${dag_node},${result_node}},\"root_id\":\"root\"}"
    push_scene "${COMPOSITION_NAME}-main" "$scene"
}

do_inference() {
    local prompt="${1:-Describe the current state of an AI agent observing a NUCLEUS composition with $INFERENCE_COUNT prior inferences.}"
    LAST_PROMPT="$prompt"
    AGENT_STATE="inferring"
    domain_render "Calling inference.complete..."
    if agent_inference_complete "$prompt" 64; then
        AGENT_STATE="ready"
        record_agent_action "inference_complete" "${LAST_INFERENCE_RESULT:0:120}" "agent"
        domain_render "Inference #$INFERENCE_COUNT complete"
    else
        AGENT_STATE="degraded"
        record_agent_action "inference_failed" "no_response" "agent"
        domain_render "Inference failed — Squirrel unavailable"
    fi
}

do_embed() {
    local text="${1:-neuralspring agent state tick=$TICK_COUNT inferences=$INFERENCE_COUNT}"
    AGENT_STATE="embedding"
    domain_render "Calling inference.embed..."
    if agent_inference_embed "$text"; then
        AGENT_STATE="ready"
        record_agent_action "inference_embed" "dim=${#LAST_EMBED_RESULT}" "agent"
        domain_render "Embedding #$EMBED_COUNT complete (${#LAST_EMBED_RESULT} chars)"
    else
        AGENT_STATE="degraded"
        record_agent_action "embed_failed" "no_response" "agent"
        domain_render "Embedding failed — Squirrel unavailable"
    fi
}

do_reason() {
    REASON_COUNT=$((REASON_COUNT + 1))
    AGENT_STATE="reasoning"
    domain_render "Reasoning step #$REASON_COUNT — inference + DAG branch..."

    local context="Agent has made $INFERENCE_COUNT inferences and $EMBED_COUNT embeddings over $TICK_COUNT ticks."
    context="$context DAG depth is ${#VERTEX_STACK[@]}."
    [[ -n "$LAST_INFERENCE_RESULT" ]] && context="$context Last result: ${LAST_INFERENCE_RESULT:0:60}"

    local prompt="Given context: $context — What should the agent do next? Choose: INFER, EMBED, or WAIT."

    if agent_inference_complete "$prompt" 32; then
        local decision="${LAST_INFERENCE_RESULT,,}"
        record_agent_action "reason" "decision=$decision" "agent"

        case "$decision" in
            *infer*) do_inference ;;
            *embed*) do_embed ;;
            *)
                AGENT_STATE="ready"
                domain_render "Reason #$REASON_COUNT: WAIT (tick $TICK_COUNT)"
                ;;
        esac
    else
        AGENT_STATE="degraded"
        record_agent_action "reason_failed" "squirrel_unavailable" "agent"
        domain_render "Reasoning failed — Squirrel unavailable"
    fi
}

domain_on_key() {
    local key="$1"
    case "$key" in
        Q|q|Escape)
            log "quit requested"
            RUNNING=false
            ;;
        I|i)
            do_inference
            ;;
        E|e)
            do_embed
            ;;
        R|r)
            do_reason
            ;;
        A|a)
            if (( AUTO_REASON_INTERVAL > 0 )); then
                AUTO_REASON_INTERVAL=0
                record_agent_action "auto_off" "disabled" "keyboard"
                domain_render "Auto-reason disabled"
            else
                AUTO_REASON_INTERVAL=20
                record_agent_action "auto_on" "interval=20" "keyboard"
                domain_render "Auto-reason enabled (every 20 ticks)"
            fi
            ;;
        *)
            log "unbound key: $key"
            dag_append_event "$COMPOSITION_NAME" "keypress" "$AGENT_STATE" \
                "[{\"key\":\"key\",\"value\":\"$key\"}]" "keyboard" "0"
            ;;
    esac
}

domain_on_click() {
    local cell="$1"
    case "$cell" in
        0) do_inference ;;
        1) do_embed ;;
        2) do_reason ;;
        *)
            log "clicked unknown target: $cell"
            ;;
    esac
    record_agent_action "click" "target=$cell" "click" "$ACCUMULATED_HOVER_MOVES"
    ACCUMULATED_HOVER_MOVES=0
}

domain_on_tick() {
    TICK_COUNT=$((TICK_COUNT + 1))
    check_proprioception

    # Closed-loop: auto-reason at interval (act → observe → adjust)
    if (( AUTO_REASON_INTERVAL > 0 && TICK_COUNT % AUTO_REASON_INTERVAL == 0 )); then
        log "auto-reason at tick $TICK_COUNT"
        do_reason
    fi
}

# ── 7. Main Loop ─────────────────────────────────────────────────────

main() {
    discover_capabilities || { err "Required primals not found (need: $REQUIRED_CAPS)"; exit 1; }

    composition_startup "neuralSpring Agent" "Agent-Driven Composition + AI Feedback Loops"

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

    # Commit game line to ledger before teardown
    if cap_available ledger && [[ -n "${SPINE_ID:-}" ]]; then
        local summary
        summary=$(printf '{"inferences":%d,"embeddings":%d,"reasons":%d,"ticks":%d,"final_state":"%s"}' \
            "$INFERENCE_COUNT" "$EMBED_COUNT" "$REASON_COUNT" "$TICK_COUNT" "$AGENT_STATE")
        ledger_append_entry "session-summary" "$summary"
        ledger_seal_spine
    fi

    composition_summary
    composition_teardown "${COMPOSITION_NAME}-main"
}

main
