#!/usr/bin/env bash
# Correctness check with real weight updates: Qwen3-0.6B full fine-tuning, followed-weight 1.0, lr 3e-6
# (learns to follow the chain), W1 K4_L256_P256, 15 steps, baseline vs fixed.
# vLLM memory utilization lowered to 0.35: full fine-tuning keeps the Adam states on the GPU during the rollout.
# Run from the repository root; baseline/fixed TRL selection as in run_w1.sh.
set -uo pipefail
R=${R:-results/issue7428}
PY=${PY:-python}  # the training environment (vLLM installed)
BASELINE=${BASELINE:-../trl-baseline-main}
[ -d "$BASELINE" ] || git worktree add --detach "$BASELINE" main || exit 1
BASELINE=$(realpath "$BASELINE")
CELL=HPML_WORK/data_prepare/out/w1/K4_L256_P256
mkdir -p $R/logs
for lab in baseline fixed; do
  trl=$([ "$lab" = baseline ] && echo "$BASELINE" || echo "$PWD")
  name=w1_K4_L256_P256_0.6B-fullft_followed1_prefix-on
  PYTHONPATH=$trl $PY HPML_WORK/benchmark_7428/bench_multiturn.py $CELL --label $lab --out-dir $R/train/$lab/$name \
    --model Qwen/Qwen3-0.6B --lora-r 0 --followed-weight 1.0 --learning-rate 3e-6 --max-steps 15 \
    --vllm-gpu-memory-utilization 0.35 > $R/logs/train_${lab}_${name}.log 2>&1
  echo "DONE $lab $name exit=$? $(grep -c '^step ' $R/logs/train_${lab}_${name}.log) steps"
done
echo ALL_DONE
