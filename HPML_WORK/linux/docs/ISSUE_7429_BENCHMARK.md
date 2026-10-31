# Linux CUDA/vLLM benchmark: issue #7428 / PR #7429

GitHub #7428 reports redundant weight synchronization between multi-turn generation calls in one rollout when colocated vLLM sleep mode is enabled. #7429 is the proposed fix.

The upstream issue's minimal logical reproduction uses CPU and a mocked vLLM; that is the upstream author's unit-test-style reproduction, not this project's experiment. Our Insomnia workflow requires a GPU: Qwen3 inference, the vLLM benchmark, and GRPO model forward/backward training all run on CUDA, and both Python programs fail if their trainer is not assigned to a CUDA device. CPU work is limited to host orchestration, reward calculation, and the toy calculator/tool-result simulation between GPU generation turns; no model training or generation falls back to CPU. The Linux benchmark requires real vLLM and the proposal's Qwen3-0.6B model, and measures synchronization/sleep counts, sync/turn/rollout latency, and peak GPU memory. The native-Windows emulation and visualization steps are in the [Windows benchmark guide](../../windows/docs/ISSUE_7429_BENCHMARK.md).

## Prepare an owned Ubuntu GPU VM

On an Ubuntu GPU VM where you have administrator access, install Git and Python before cloning the repository or running the setup script:

```bash
apt update
apt install git -y
apt install software-properties-common -y
apt install build-essential -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.13 python3.13-full python3.13-dev -y
```

These commands install Git, a C/C++ toolchain and Python development headers required by Triton, and Python, not the NVIDIA driver. Confirm the VM's NVIDIA driver is installed and the GPU is visible with `nvidia-smi`; the Linux setup script currently supports driver-reported CUDA versions 12.9 and 13.0. Then clone or check out this repository, go to its root, and follow the setup and benchmark instructions below. The setup script detects Python 3.13 automatically and creates `.venv-issue7429`. If the C compiler is missing, the script installs `build-essential` when run as root. It also installs the matching `pythonX.Y-dev` package if Python headers are missing. On a managed cluster, load a compiler module and a Python module with development headers, or set `CC` to an available compiler. For Qwen3 download, it defaults to `https://hf-mirror.com`, which may be more reachable from mainland China, and disables the Xet transfer backend because the mirror may not authorize Xet CAS requests. Hugging Face Hub then downloads through its regular HTTP path. Set `HF_ENDPOINT` to another Hugging Face-compatible endpoint before running setup to override it.

## Run interactively on Insomnia

Use an interactive Slurm allocation and run the benchmark in that allocation. There is no need to submit a second `sbatch` job for the benchmark. Request the compute node first from the login node:

```bash
srun -A edu --partition=short --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=08:00:00 --pty /bin/bash
```

Once Slurm assigns a node, go to the repository root and prepare the environment and model. This setup script is intended to run on the allocated GPU node: it uses the node's driver to select matching CUDA 12.9 or CUDA 13.0 PyTorch, vLLM 0.32.0, and TorchCodec 0.16.0 wheels; installs the project dependencies into `.venv-issue7429`; verifies the GPU/library imports; and downloads Qwen3-0.6B.

```bash
cd ~/trl
module load anaconda/2023.09
bash HPML_WORK/linux/scripts/setup_issue7429_linux.sh
source .venv-issue7429/bin/activate
```

The Anaconda module supplies the base Python interpreter (3.11 or newer); the setup script creates the project virtual environment and installs Python packages into it. You do not need to create a separate Conda environment. A virtual environment uses its base Python interpreter rather than bundling another Python installation. If you already have a compatible Python 3.11+ active, the setup script can use that instead.

The venv and downloaded model live in your repository under shared storage, so they normally persist when you request a different compute node. Activate the venv and confirm CUDA access on each allocation. If the new node has a different CUDA driver version, rerun setup there to rebuild the venv with matching wheels.

## Run the benchmark

This script measures **only the checked-out TRL source**; it does not switch between or emulate the affected and fixed implementations. Run it once in the affected baseline checkout and once in the checkout containing the fix. Each run's JSON records the observed sync/sleep counts and source commit.

```bash
source .venv-issue7429/bin/activate
python HPML_WORK/linux/scripts/issue7429_live_benchmark.py \
  --label baseline \
  --turns 2 \
  --iterations 5 \
  --dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.3 \
  --output HPML_WORK/linux/results/baseline.json
```

Repeat the same command in the fixed checkout, changing only `--label fixed` and the output filename to `fixed.json`. The model defaults to `models/Qwen3-0.6B` under the repository root; the model path, revision, GPU, and all benchmark settings must match between runs. Lower vLLM memory utilization if colocated allocation leaves too little room, and use the same value in both runs.

For two turns, the expected observations from the **two separate source revisions** are:

| Separate run against | Weight syncs per rollout | Sleep calls inside rollout | Sleep after rollout |
|---|---:|---:|---:|
| Affected baseline checkout | 2 | 2 | 0 |
| Fixed checkout | 1 | 0 | 1 |

The table is not two policies exercised by one benchmark invocation: each row comes from running the same script against a different checkout. Zero sleeps **inside** the fixed rollout does not mean sleep mode was disabled; the engine sleeps once after the complete rollout, before scoring/training needs its GPU memory. `behavior` in each JSON should be `buggy` for the affected checkout and `fixed` for the fixed checkout; `source_commit` identifies exactly what was measured. The harness stops at the pre-scoring boundary so the issue-specific measurement isolates inter-turn behavior and skips an optimizer step. For end-to-end project metrics, plug in the shared chain-arithmetic dataset/environment and record accuracy/reward as well.

## Run the optional GRPO training workload

The issue benchmark above measures generation behavior and does not perform an optimizer step. To run the separate GPU LoRA/GRPO integration workload within the same interactive allocation:

```bash
python HPML_WORK/linux/scripts/train_grpo_benchmark.py \
  --model models/Qwen3-0.6B \
  --max-steps 10 \
  --dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.3 \
  --output-dir HPML_WORK/linux/results/interactive-training \
  --metrics-output HPML_WORK/linux/results/interactive-training.json
```

This is a short synthetic arithmetic workload, not the final multi-turn calculator/GSM8K experiment. Results are saved in the paths passed above.

Visualize its training metrics separately from the rollout benchmark:

```bash
python HPML_WORK/shared/scripts/visualize_benchmark.py \
  --input HPML_WORK/linux/results/interactive-training.json \
  --output HPML_WORK/linux/results/interactive-training.png
```

The training plot shows total training time, median step time, per-step times, peak allocated/reserved GPU memory, and final loss. Multiple training JSON files can be passed to `--input` to compare runs; use separate commands for training and rollout reports.

## Visualize and compare

Save separate baseline and fixed JSON results using `--output`, then create one comparison plot:

```bash
python HPML_WORK/shared/scripts/visualize_benchmark.py \
  --input HPML_WORK/linux/results/baseline.json HPML_WORK/linux/results/fixed.json \
  --output HPML_WORK/linux/results/comparison.png
```

The figure compares rollout time, weight transfer time, per-turn generation latency, and peak allocated VRAM. Keep hardware, model revision, software, workload, generated token counts, and all CLI arguments matched; repeat runs and report medians/spread.
