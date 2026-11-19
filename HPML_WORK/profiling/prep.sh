#!/usr/bin/env bash

cd /workspace/trl

set -euo pipefail

export HPML_REPO="$PWD"
export HPML_WORK_DIR="$HPML_REPO/HPML_WORK"
export HPML_DATA_DIR="$HPML_WORK_DIR/data_prepare"
export PYTHON=python3.12

git fetch origin
git switch -c ann-run-baseline d5c8d4c38c17d38de8938eed3c1d3a1df5298a91

export HPML_RUN_DIR="$HPML_WORK_DIR/linux/results/manual-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HPML_RUN_DIR"/{baseline,fixed}

export HF_ENDPOINT=https://huggingface.co

# Training environment setup
export HPML_CUDA_VERSION=13.0

bash "$HPML_WORK_DIR/linux/scripts/setup_issue7429_linux.sh"
source "$HPML_REPO/.venv-issue7429/bin/activate"

python -m pip install \
  "transformers==5.19.0" "peft==0.21.2" "bm25s==0.3.16"
python -m pip check
python -m pip freeze > "$HPML_RUN_DIR/environment.txt"

deactivate

# Data preparation
"$PYTHON" -m venv "$HPML_REPO/.venv-hpml"
source "$HPML_REPO/.venv-hpml/bin/activate"

bash "$HPML_DATA_DIR/generate_all.sh" --install

deactivate

source "$HPML_REPO/.venv-issue7429/bin/activate"

