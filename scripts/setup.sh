#!/usr/bin/env bash
# FreeForward4D portable environment setup.
#
# Installs PyTorch 2.10.0 + torchvision 0.25.0 from the official cu128 wheel index,
# then requirements.txt, then the freeforward package in editable mode. No model
# weights are downloaded unless --with-priors is given.
#
# Raw Waymo TFRecord conversion is NOT handled here: TF 2.12 needs NumPy<1.24, which
# conflicts with the training environment. Use a separate Python 3.10 conda env:
#   conda create -n freeforward-data python=3.10
#   pip install -r requirements-data.txt
set -euo pipefail
FREEFORWARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FREEFORWARD_PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
WITH_PRIORS=0
SKIP_CHECK=0
for arg in "$@"; do
    case "$arg" in
        --with-priors) WITH_PRIORS=1 ;;
        --skip-check) SKIP_CHECK=1 ;;
        -h|--help)
            echo "usage: scripts/setup.sh [--with-priors] [--skip-check]"
            echo "  PYTHON env var selects the interpreter (default: python)."
            exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "== FreeForward4D setup (root: $FREEFORWARD_PROJECT_ROOT, python: $PYTHON)"

"$PYTHON" - <<'PY'
import sys
v = sys.version_info
if v < (3, 10):
    raise SystemExit("Python >= 3.10 required, found %d.%d" % (v.major, v.minor))
if (v.major, v.minor) != (3, 10):
    print("note: Python 3.10 is the recommended version; found %d.%d" % (v.major, v.minor))
PY

echo "== torch 2.10.0 + torchvision 0.25.0 from the official cu128 index"
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0" "torchvision==0.25.0"

echo "== runtime requirements"
"$PYTHON" -m pip install -r requirements.txt

echo "== freeforward package (editable; deps already pinned above)"
"$PYTHON" -m pip install -e . --no-deps

if [ "$WITH_PRIORS" -eq 1 ]; then
    echo "== downloading prior model weights"
    "$PYTHON" scripts/download_priors.py
fi

if [ "$SKIP_CHECK" -eq 0 ]; then
    echo "== environment check"
    "$PYTHON" scripts/check_environment.py
fi

echo "== setup complete"
