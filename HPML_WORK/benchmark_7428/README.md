# Benchmark for the issue #7428 fix

These scripts measure the fix for [huggingface/trl#7428](https://github.com/huggingface/trl/issues/7428). The bug: with `vllm_mode="colocate"` and `vllm_enable_sleep_mode=True`, every `generate()` call put vLLM to level-2 sleep. A rollout with K tool calls therefore synced the same weights K + 1 times. The scripts run the TRL of `main` (baseline) and of this branch (fixed) on the same multi-turn GRPO workloads and record where the time goes.

Everything is run from the repository root, on this branch.

## How baseline and fixed are selected

The repository never switches branches. On first use, the run scripts create a git worktree of `main` next to the repository (`../trl-baseline-main`; override with `BASELINE=<path>`). Each run then picks the TRL under test through `PYTHONPATH`:

| run | `PYTHONPATH` | TRL imported |
|---|---|---|
| baseline | `../trl-baseline-main` | `main` (the per-turn sleep and reload) |
| fixed | the repository root | this branch |

`PYTHONPATH` takes precedence over an editable install of TRL. Each run records the path and commit of the TRL it actually imported, so you can check which code ran:
- `bench_multiturn.py` writes them to `summary.json` (`trl_path`, `git_commit`);
- `run_live.sh` adds them to each live report (`trl_path`, `trl_git_commit`).

You can also check from the results: a baseline step syncs once per generation turn, a fixed step once.

The worktree stays at the commit of `main` it was created from. To move it to the current `main`, run `git -C ../trl-baseline-main checkout --detach main`.

## Files

| File | Purpose |
|---|---|
| `bench_multiturn.py` | Runs a few GRPO steps on one W1/W2 cell (built on `../data_prepare/smoke_test.py`) and records every engine call with timestamps: `sync_weights`, `llm.generate` (with prompt / prefix-cached / completion tokens), `llm.sleep`, `llm.wake_up`, the rollout, the generate+score phase and the step boundaries. It also samples NVML GPU memory every 50 ms. |
| `run_live.sh` | Rollout-only benchmark (`../linux/scripts/issue7429_live_benchmark.py`): Qwen3-0.6B + LoRA, 2/4/8 turns, 10 iterations, baseline vs fixed (6 runs). |
| `run_w1.sh` | W1 main experiment: Qwen3-1.7B + LoRA r16, K ∈ {2, 4, 8}, L_obs = 256, P = 256, prefix caching on/off, baseline vs fixed (12 runs). |
| `run_w2.sh` | W2 supplement: `K3_top1` and `K3_top3`, prefix caching on (4 runs). |
| `run_correctness.sh` | Check with real weight updates: Qwen3-0.6B full fine-tuning, chain-following reward, W1 K = 4, 15 steps (2 runs). |
| `aggregate.py` | Turns the run folders into CSV tables. |
| `plot.py` | Turns the tables into figures. |

Other files used:
- `../data_prepare/` (from Yidan's branch, unchanged): W1/W2/W3 generators, environments and checks.
- `../linux/scripts/issue7429_live_benchmark.py` (from Yuyun's branch, unchanged).

## Setup

Use two environments outside the repository, because the dataset token counts depend on the tokenizer version pinned in `data_prepare/requirements.txt`:

```bash
# Data generation (CPU only)
uv venv --python 3.12 ~/venvs/datagen
VIRTUAL_ENV=~/venvs/datagen uv pip install -r HPML_WORK/data_prepare/requirements.txt

# Training (GPU)
uv venv --python 3.12 ~/venvs/train
VIRTUAL_ENV=~/venvs/train uv pip install "vllm==0.32.0" -e ".[test,peft]" matplotlib
```

Results go to `results/issue7428/`, which is not meant to be committed. Keep git from listing it:

```bash
echo 'results/' >> .git/info/exclude
```

Generate and check the data. It is seeded, so you always get the same data, written to `HPML_WORK/data_prepare/out/` (git-ignored, about 250 MB):

```bash
cd HPML_WORK/data_prepare
P=~/venvs/datagen/bin/python
$P w1_generate.py && $P w1_check.py && $P w2_kb.py && $P w2_generate.py && $P w2_check.py
cd ../..
```

## Run

```bash
export PY=~/venvs/train/bin/python
bash HPML_WORK/benchmark_7428/run_live.sh
bash HPML_WORK/benchmark_7428/run_w1.sh
bash HPML_WORK/benchmark_7428/run_w2.sh
bash HPML_WORK/benchmark_7428/run_correctness.sh

# Tables and figures
$PY HPML_WORK/benchmark_7428/aggregate.py
$PY HPML_WORK/benchmark_7428/plot.py
```

- Each run script prints one `DONE ... exit=<code> <n> steps` line per run, and the run's output goes to `results/issue7428/logs/`.
- `R=<dir>` writes the results somewhere other than `results/issue7428`.
- `aggregate.py` and `plot.py` take the same directory as `--results-dir`.
- Running time on one RTX 4090: about 1.5 h for everything. A K = 8 run takes about 5 min, including model loading.

A single run (fixed; for the baseline, use `PYTHONPATH=../trl-baseline-main` and `--label baseline`):

```bash
PYTHONPATH=$PWD $PY HPML_WORK/benchmark_7428/bench_multiturn.py HPML_WORK/data_prepare/out/w1/K8_L256_P256 \
    --label fixed --out-dir results/issue7428/train/fixed/my_run --model Qwen/Qwen3-1.7B --max-steps 10
```

- `--disable-prefix-caching` turns off vLLM's automatic prefix caching.
- `--lora-r 0` trains all parameters.
- `--followed-weight` gives the chain-following reward a non-zero weight.
- `--out-dir` should be a new directory: the files of an existing run are overwritten.

## Outputs

Each run writes to `results/issue7428/train/<baseline|fixed>/<config>/`:

| File | Contents |
|---|---|
| `events.jsonl` | One line per engine call or phase: `event`, `step`, `t_start` (s), `ms`, plus token counts for `llm.generate`, `level` for `llm.sleep` and `tags` for `llm.wake_up`. |
| `gpu_mem.csv` | `t_s,used_mib`: whole-device memory (NVML), on the same clock as `events.jsonl`. |
| `steps.json` | Per-step aggregates: step / rollout / sync / sleep / wake-up / generate times, turns, per-turn tokens, reward, tool calls. |
| `log_history.json` | The trainer's `log_history`. |
| `summary.json` | TRL path, commit and branch, versions, arguments, wall time, peak memory. |

The live benchmark writes `results/issue7428/live/<baseline|fixed>/turns<N>.json`.

`aggregate.py` writes these tables to `results/issue7428/tables/` (step 0 is warmup and is excluded from medians):

| File | One row per |
|---|---|
| `runs.csv` | run |
| `steps.csv` | step |
| `turns.csv` | generation turn |
| `comparison.csv` | configuration, with baseline and fixed side by side and the speedups |
| `live.csv`, `live_summary.csv` | live benchmark iteration / configuration |

`plot.py` writes PNG and PDF figures to `results/issue7428/figures/`.

`sync_weights()` calls `llm.wake_up(tags=["weights"])` itself, so that wake-up is already included in the sync time. The tables split wake-ups by tag (`wake_weights_ms`, `wake_kv_cache_ms`) so the step breakdown doesn't count it twice.

## Notes

- **Model choice.** Qwen3-1.7B follows the W1 chain in ≥ 96% of rollouts, so both sides see the same number of turns, which the timing comparison needs. Its answers are almost never correct, so the W1 answer reward is constant within each group and the weights don't change (`grad_norm` = 0). That doesn't affect the timing, because every sync pushes the full weights. `run_correctness.sh` covers training with real weight updates.
- **Long-context cells don't fit.** On 24 GB, cells with L_obs = 1024 or K = 16 likely don't fit and were not run. Qwen3-4B + LoRA needs a 48 GB GPU (see `../data_prepare/README.md`).
- **Peak memory.** `torch.cuda.max_memory_allocated` is unreliable under sleep mode, so use NVML (`gpu_mem.csv`).
- **Exit message.** vLLM prints `CUDA Error: invalid argument at cumem_allocator.cpp` at process exit in sleep mode. It is harmless: the exit code is 0.
- **Dirty flag.** `summary.json` marks a run `git_dirty` when the imported TRL checkout has uncommitted changes to tracked files, including staged files.
