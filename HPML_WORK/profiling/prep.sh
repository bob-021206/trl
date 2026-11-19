#!/usr/bin/env bash
set -euo pipefail

# Prepare the current checkout. Older baseline commits do not contain this harness.
export HPML_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HPML_REPO"

export HPML_WORK_DIR="$HPML_REPO/HPML_WORK"
export HPML_DATA_DIR="$HPML_WORK_DIR/data_prepare"
export PYTHON=python3.12

export HPML_RUN_DIR="$HPML_WORK_DIR/linux/results/manual-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HPML_RUN_DIR"/{baseline,fixed}

export HF_ENDPOINT=https://huggingface.co

# Training environment setup
export HPML_CUDA_VERSION=13.0

bash "$HPML_WORK_DIR/linux/scripts/setup_issue7429_linux.sh"
source "$HPML_REPO/.venv-issue7429/bin/activate"

python -m pip install \
  "transformers==5.19.0" "peft==0.21.2" "bm25s==0.3.16"
python -m pip install -r "$HPML_WORK_DIR/profiling/requirements.txt"
python -m pip check
python -m pip freeze > "$HPML_RUN_DIR/environment.txt"

deactivate

# Data preparation
"$PYTHON" -m venv "$HPML_REPO/.venv-hpml"
source "$HPML_REPO/.venv-hpml/bin/activate"

bash "$HPML_DATA_DIR/generate_all.sh" --install

deactivate

source "$HPML_REPO/.venv-issue7429/bin/activate"

# When invoked with bash, activation above applies only to this script's process.
printf '\nPreparation complete. In your terminal, run:\n  cd "%s"\n  source .venv-issue7429/bin/activate\n  python -m HPML_WORK.profiling.run --dry-run\n' "$HPML_REPO"

