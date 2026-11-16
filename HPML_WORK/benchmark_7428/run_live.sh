#!/usr/bin/env bash
# Live benchmark (rollout only, ../linux/scripts/issue7429_live_benchmark.py): Qwen3-0.6B + LoRA, 2/4/8 turns,
# 10 iterations each, baseline vs fixed. Run from the repository root; baseline/fixed TRL selection as in run_w1.sh.
# The live benchmark records the commit of this repository as `source_commit`, so the commit of the TRL actually
# imported is added to each report as `trl_path` / `trl_git_commit`. Its `behavior` field ("buggy" / "fixed") also
# shows which code ran.
set -uo pipefail
R=${R:-$PWD/results/issue7428}
PY=${PY:-python}  # the training environment (vLLM installed)
BASELINE=${BASELINE:-../trl-baseline-main}
[ -d "$BASELINE" ] || git worktree add --detach "$BASELINE" main || exit 1
BASELINE=$(realpath "$BASELINE")
mkdir -p $R/logs
for lab in baseline fixed; do
  trl=$([ "$lab" = baseline ] && echo "$BASELINE" || echo "$PWD")
  for t in 2 4 8; do
    out=$R/live/$lab/turns$t.json
    PYTHONPATH=$trl $PY HPML_WORK/linux/scripts/issue7429_live_benchmark.py --model Qwen/Qwen3-0.6B --turns $t \
      --iterations 10 --label $lab --output $out > $R/logs/live_${lab}_turns$t.log 2>&1
    status=$?
    [ -f "$out" ] && $PY - "$out" "$trl" "$(git -C "$trl" rev-parse HEAD)" <<'EOF'
import json, sys
path, trl_path, commit = sys.argv[1:]
report = json.load(open(path))
report.update(trl_path=trl_path, trl_git_commit=commit)
json.dump(report, open(path, "w"), indent=2)
EOF
    echo "DONE $lab turns=$t exit=$status $(grep -o 'observed behavior: [a-z]*' $R/logs/live_${lab}_turns$t.log)"
  done
done
echo ALL_DONE
