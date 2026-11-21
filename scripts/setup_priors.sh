#!/usr/bin/env bash
set -euo pipefail
FREEFORWARD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FREEFORWARD_PROJECT_ROOT"
mkdir -p third_party
checkout() {
  local url="$1" directory="$2" revision="$3"
  if [[ ! -d "$directory/.git" ]]; then
    if [[ -e "$directory" ]]; then
      printf 'Directory exists without Git metadata: %s\n' "$directory" >&2
      exit 2
    fi
    git clone --filter=blob:none --no-checkout "$url" "$directory"
  fi
  if ! git -C "$directory" diff --quiet; then
    printf 'Preserve your local changes before updating %s\n' "$directory" >&2
    exit 2
  fi
  git -C "$directory" fetch --depth 1 origin "$revision"
  git -C "$directory" checkout --detach "$revision"
}
checkout https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
checkout https://github.com/microsoft/MoGe.git third_party/MoGe b942f00bdc2a2a23ebb474fbe034d487e6dcceec
checkout https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2 b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
"${PYTHON:-python}" -m pip install -c requirements.txt -r requirements-priors.txt
export MAX_JOBS="${MAX_JOBS:-4}"
# Import MoGe-2 directly from its pinned source: its old full setup references
# obsolete Git dependencies and unrelated UI/training extras. Inference uses
# requirements-priors.txt and the licensed local utils3d compatibility module.
"${PYTHON:-python}" -m pip install --no-deps --no-build-isolation -e third_party/Grounded-SAM-2
"${PYTHON:-python}" -m pip install --no-deps -e third_party/vggt-omega
printf 'Foundation-model source dependencies are ready. Run scripts/download_priors.py next.\n'
