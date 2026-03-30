#!/bin/bash
# ============================================================================
# Three-way Sync: Local <-> Git <-> Server
# Usage:
#   bash sync.sh push       # Local -> Git + Server
#   bash sync.sh pull       # Server -> Local + Git
#   bash sync.sh to-server  # Only sync local -> server
#   bash sync.sh status     # Show sync status
# ============================================================================
set -e

SERVER="szs_cpu"
SERVER_DIR="/data/szs/250010072/nwh/nips-modelsteal"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

action="${1:-status}"

scp_to_server() {
    echo "  Syncing local -> server via scp..."
    cd "${LOCAL_DIR}"
    local dirs_to_sync="configs scripts src results"
    local files_to_sync="*.sh *.txt *.yaml *.yml *.md *.py LICENSE .gitignore"
    for f in ${files_to_sync}; do
        if ls ${f} 1>/dev/null 2>&1; then
            scp ${f} ${SERVER}:${SERVER_DIR}/ 2>/dev/null || true
        fi
    done
    for d in ${dirs_to_sync}; do
        if [ -d "${d}" ]; then
            ssh ${SERVER} "mkdir -p ${SERVER_DIR}/${d}"
            scp -r ${d}/* ${SERVER}:${SERVER_DIR}/${d}/ 2>/dev/null || true
        fi
    done
    echo "  scp sync complete."
}

scp_from_server() {
    echo "  Syncing server -> local via scp..."
    cd "${LOCAL_DIR}"
    local dirs_to_sync="configs scripts src results"
    for d in ${dirs_to_sync}; do
        ssh ${SERVER} "ls ${SERVER_DIR}/${d}/ 2>/dev/null" && {
            mkdir -p "${d}"
            scp -r ${SERVER}:${SERVER_DIR}/${d}/* ${d}/ 2>/dev/null || true
        }
    done
    for ext in sh txt yaml yml md py; do
        scp ${SERVER}:${SERVER_DIR}/*.${ext} . 2>/dev/null || true
    done
    echo "  scp sync complete."
}

sync_push() {
    echo "=== [1/3] Committing local changes ==="
    cd "${LOCAL_DIR}"
    git add -A
    if git diff --cached --quiet; then
        echo "  No local changes to commit."
    else
        git commit --no-verify -m "sync: $(date +%Y%m%d_%H%M%S)"
    fi
    echo "=== [2/3] Pushing to Git remote ==="
    git push origin main
    echo "=== [3/3] Syncing to server ==="
    scp_to_server
    echo "=== Push sync complete ==="
}

sync_status() {
    echo "=== Local Status ==="
    cd "${LOCAL_DIR}"
    git log --oneline -3
    git status -sb
    echo ""
    echo "=== Server Status ==="
    ssh ${SERVER} "ls -la ${SERVER_DIR}/*.sh ${SERVER_DIR}/configs/ 2>/dev/null | head -20"
    echo "  Server path: ${SERVER_DIR}"
}

case "$action" in
    push)        sync_push ;;
    pull)        scp_from_server ;;
    status)      sync_status ;;
    to-server)   scp_to_server ;;
    from-server) scp_from_server ;;
    *)
        echo "Usage: bash sync.sh {push|pull|status|to-server|from-server}"
        exit 1
        ;;
esac
