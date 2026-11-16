#!/usr/bin/env bash
# Generate and check every dataset (W1, W2, W3) into HPML_WORK/data/out/. CPU only.
#
#   bash HPML_WORK/data/generate_all.sh             # generate + check, using the current python
#   bash HPML_WORK/data/generate_all.sh --install   # first pip install the pinned requirements.txt
#
# Everything is seeded, so re-running gives identical data (the checks compare sha256 with meta.json).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [[ "${1:-}" == "--install" ]]; then
    pip install -r requirements.txt
fi

echo "== W1: chain arithmetic (24 cells) =="
python w1_generate.py
python w1_check.py

echo "== W2: knowledge base + multi-hop retrieval (12 cells) =="
python w2_kb.py
python w2_generate.py
python w2_check.py

echo "== W3: MuSiQue (3 cells) =="
python w3_generate.py
python w3_check.py

echo "Done: all datasets are in $DIR/out"
du -sh out/w1 out/w2 out/w3
