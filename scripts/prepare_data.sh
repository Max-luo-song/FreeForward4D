#!/usr/bin/env bash
set -euo pipefail
FREEFORWARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FREEFORWARD_PROJECT_ROOT"
exec "${PYTHON:-python}" scripts/prepare_data.py "$@"
