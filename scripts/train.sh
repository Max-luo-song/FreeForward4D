#!/usr/bin/env bash
# FreeForward4D training launcher wrapper. Examples:
#   scripts/train.sh --stage 2s --dry-run                 # print resolved plan
#   scripts/train.sh --stage 2s                           # 8x H200, single node
set -euo pipefail
FREEFORWARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FREEFORWARD_PROJECT_ROOT"
exec "${PYTHON:-python}" scripts/launch_training.py "$@"
