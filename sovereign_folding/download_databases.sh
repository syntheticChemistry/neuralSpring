#!/usr/bin/env bash
#
# Sovereign Folding — MSA Database Download Script
#
# Downloads sequence databases required for protein/RNA/DNA structure prediction.
# Supports resuming interrupted downloads (wget -c).
#
# Usage:
#   ./download_databases.sh phase1   # UniRef90 + PDB (~300 GB)
#   ./download_databases.sh phase2   # + Rfam + RNAcentral (~55 GB)
#   ./download_databases.sh phase3   # + BFD (~1.7 TB)
#   ./download_databases.sh all      # Everything
#
# Storage requirement: ~3 TB for all phases

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR="${FOLD_DATA_DIR:-$HOME/fold_databases}"
LOG_FILE="${BASE_DIR}/download.log"

mkdir -p "$BASE_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_space() {
    local dir="$1"
    local needed_gb="$2"
    local available_gb
    available_gb=$(df -BG "$dir" | tail -1 | awk '{print $4}' | tr -d 'G')
    if (( available_gb < needed_gb )); then
        log "ERROR: Need ${needed_gb}GB free, only ${available_gb}GB available in $dir"
        return 1
    fi
    log "Storage OK: ${available_gb}GB available, need ${needed_gb}GB"
}

download_with_resume() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        log "SKIP: $desc already exists at $dest"
        return 0
    fi

    log "DOWNLOADING: $desc"
    log "  URL: $url"
    log "  Dest: $dest"

    wget -c --progress=dot:giga -O "$dest" "$url" 2>&1 | tee -a "$LOG_FILE"

    log "DONE: $desc ($(du -sh "$dest" | cut -f1))"
}

# ─── Phase 1: UniRef90 + PDB Templates (~300 GB) ────────────────────────────

download_phase1() {
    log "=== PHASE 1: UniRef90 + PDB Templates ==="
    check_space "$BASE_DIR" 350

    local uniref_dir="$BASE_DIR/uniref90"
    mkdir -p "$uniref_dir"
    download_with_resume \
        "ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz" \
        "$uniref_dir/uniref90.fasta.gz" \
        "UniRef90 (~100 GB)"

    local pdb_dir="$BASE_DIR/pdb_mmcif"
    mkdir -p "$pdb_dir"
    download_with_resume \
        "https://files.wwpdb.org/pub/pdb/data/structures/all/mmCIF/mmcif_files.tar.gz" \
        "$pdb_dir/mmcif_files.tar.gz" \
        "PDB mmCIF structures (~60 GB compressed)"

    local pdb70_dir="$BASE_DIR/pdb70"
    mkdir -p "$pdb70_dir"
    download_with_resume \
        "https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pdb70_from_mmcif_latest.tar.gz" \
        "$pdb70_dir/pdb70_from_mmcif_latest.tar.gz" \
        "PDB70 HHsuite database (~60 GB)"

    log "=== PHASE 1 COMPLETE ==="
}

# ─── Phase 2: Rfam + RNAcentral (~55 GB) ────────────────────────────────────

download_phase2() {
    log "=== PHASE 2: Rfam + RNAcentral ==="
    check_space "$BASE_DIR" 60

    local rfam_dir="$BASE_DIR/rfam"
    mkdir -p "$rfam_dir"
    download_with_resume \
        "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz" \
        "$rfam_dir/Rfam.cm.gz" \
        "Rfam covariance models (~1 GB)"
    download_with_resume \
        "ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full.gz" \
        "$rfam_dir/Rfam.full.gz" \
        "Rfam full alignments (~4 GB)"

    local rnacentral_dir="$BASE_DIR/rnacentral"
    mkdir -p "$rnacentral_dir"
    download_with_resume \
        "ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz" \
        "$rnacentral_dir/rnacentral_active.fasta.gz" \
        "RNAcentral active sequences (~50 GB)"

    log "=== PHASE 2 COMPLETE ==="
}

# ─── Phase 3: BFD (~1.7 TB) ─────────────────────────────────────────────────

download_phase3() {
    log "=== PHASE 3: BFD (Big Fantastic Database) ==="
    check_space "$BASE_DIR" 1800

    local bfd_dir="$BASE_DIR/bfd"
    mkdir -p "$bfd_dir"
    download_with_resume \
        "https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz" \
        "$bfd_dir/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz" \
        "BFD metaclust (~1.7 TB)"

    log "=== PHASE 3 COMPLETE ==="
}

# ─── Index Databases ─────────────────────────────────────────────────────────

index_databases() {
    log "=== INDEXING DATABASES ==="

    if command -v mmseqs &>/dev/null; then
        if [ -f "$BASE_DIR/uniref90/uniref90.fasta.gz" ]; then
            log "Indexing UniRef90 with MMseqs2..."
            local idx_dir="$BASE_DIR/uniref90/mmseqs_index"
            mkdir -p "$idx_dir"
            mmseqs createdb "$BASE_DIR/uniref90/uniref90.fasta.gz" "$idx_dir/uniref90_db"
            mmseqs createindex "$idx_dir/uniref90_db" "$idx_dir/tmp"
            log "UniRef90 index complete"
        fi
    else
        log "WARN: mmseqs2 not installed. Install with: apt install mmseqs2"
    fi

    log "=== INDEXING COMPLETE ==="
}

# ─── Main ────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    phase1)
        download_phase1
        ;;
    phase2)
        download_phase1
        download_phase2
        ;;
    phase3)
        download_phase1
        download_phase2
        download_phase3
        ;;
    all)
        download_phase1
        download_phase2
        download_phase3
        ;;
    index)
        index_databases
        ;;
    status)
        log "=== DATABASE STATUS ==="
        for db in uniref90 pdb_mmcif pdb70 rfam rnacentral bfd; do
            local dir="$BASE_DIR/$db"
            if [ -d "$dir" ]; then
                size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                files=$(find "$dir" -type f 2>/dev/null | wc -l)
                log "  $db: $size ($files files)"
            else
                log "  $db: NOT DOWNLOADED"
            fi
        done
        ;;
    help|*)
        echo "Usage: $0 {phase1|phase2|phase3|all|index|status}"
        echo ""
        echo "  phase1  - UniRef90 + PDB (~300 GB)"
        echo "  phase2  - + Rfam + RNAcentral (~55 GB)"
        echo "  phase3  - + BFD (~1.7 TB)"
        echo "  all     - Everything (~2.1 TB)"
        echo "  index   - Build search indices"
        echo "  status  - Show download status"
        echo ""
        echo "Set FOLD_DATA_DIR to change download location (default: ~/fold_databases)"
        ;;
esac
