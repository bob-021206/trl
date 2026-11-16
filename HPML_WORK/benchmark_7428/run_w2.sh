#!/usr/bin/env bash
# W2 supplement: Qwen3-1.7B + LoRA r16, 10 steps, K3_top1 and K3_top3, prefix caching on, baseline vs fixed.
# Run from the repository root; baseline/fixed TRL selection as in run_w1.sh.
set -uo pipefail
R=${R:-results/issue7428}
PY=${PY:-python}  # the training environment (vLLM installed)
BASELINE=${BASELINE:-../trl-baseline-main}
[ -d "$BASELINE" ] || git worktree add --detach "$BASELINE" main || exit 1
BASELINE=$(realpath "$BASELINE")
DATA=HPML_WORK/data_prepare/out/w2
mkdir -p $R/logs
for cell in K3_top1 K3_top3; do
  for lab in baseline fixed; do
    trl=$([ "$lab" = baseline ] && echo "$BASELINE" || echo "$PWD")
    name=w2_${cell}_1.7B_prefix-on
    PYTHONPATH=$trl $PY HPML_WORK/benchmark_7428/bench_multiturn.py $DATA/$cell --label $lab \
      --out-dir $R/train/$lab/$name --model Qwen/Qwen3-1.7B --max-steps 10 \
      > $R/logs/train_${lab}_${name}.log 2>&1
    echo "DONE $lab $name exit=$? $(grep -c '^step ' $R/logs/train_${lab}_${name}.log) steps"
  done
done
echo ALL_DONE
