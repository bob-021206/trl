#!/usr/bin/env bash
# W1 main experiment: Qwen3-1.7B + LoRA r16, 10 steps, P=256, L_obs=256, K in {2,4,8},
# baseline (main) vs fixed (this branch), prefix caching on/off.
# Run from the repository root. The repository stays on its branch: the baseline imports TRL from a git worktree of
# `main` (created on first use) through PYTHONPATH, the fixed runs import TRL from this repository.
set -uo pipefail
R=${R:-results/issue7428}
PY=${PY:-python}  # the training environment (vLLM installed)
BASELINE=${BASELINE:-../trl-baseline-main}
[ -d "$BASELINE" ] || git worktree add --detach "$BASELINE" main || exit 1
BASELINE=$(realpath "$BASELINE")
DATA=HPML_WORK/data_prepare/out/w1
mkdir -p $R/logs
for K in 2 4 8; do
  for pc in on off; do
    for lab in baseline fixed; do
      trl=$([ "$lab" = baseline ] && echo "$BASELINE" || echo "$PWD")
      flag=$([ "$pc" = off ] && echo --disable-prefix-caching || echo "")
      name=w1_K${K}_L256_P256_1.7B_prefix-${pc}
      PYTHONPATH=$trl $PY HPML_WORK/benchmark_7428/bench_multiturn.py $DATA/K${K}_L256_P256 --label $lab $flag \
        --out-dir $R/train/$lab/$name --model Qwen/Qwen3-1.7B --max-steps 10 \
        > $R/logs/train_${lab}_${name}.log 2>&1
      echo "DONE $lab $name exit=$? $(grep -c '^step ' $R/logs/train_${lab}_${name}.log) steps"
    done
  done
done
echo ALL_DONE
