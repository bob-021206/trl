#!/usr/bin/env bash
#SBATCH --job-name=hpml-grpo-benchmark
#SBATCH --account=edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=HPML_WORK/linux/results/insomnia-%j.out
#SBATCH --error=HPML_WORK/linux/results/insomnia-%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

module load anaconda/2023.09
eval "$(conda shell.bash hook)"
conda_env="${HPML_CONDA_ENV:-$HOME/.conda/envs/trl-issue7429}"
if [[ ! -x "$conda_env/bin/python" ]]; then
    echo "Conda environment not found: $conda_env" >&2
    echo "Create it with: conda create --prefix \"$conda_env\" python=3.12 -y" >&2
    exit 1
fi
conda activate "$conda_env"
python -c 'import sys; assert sys.version_info >= (3, 11), f"Python 3.11 or newer is required, got {sys.version.split()[0]}"'

venv="${HPML_VENV:-.venv-issue7429}"
if [[ ! -f "$venv/bin/activate" ]]; then
    echo "Python virtual environment not found: $venv" >&2
    echo "Run HPML_WORK/linux/scripts/setup_issue7429_linux.sh on the cluster first." >&2
    exit 1
fi
source "$venv/bin/activate"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; this job requires a GPU node." >&2
    exit 1
fi
nvidia-smi

job_id="${SLURM_JOB_ID:-manual}"
benchmark_output="${HPML_BENCHMARK_OUTPUT:-HPML_WORK/linux/results/insomnia-${job_id}-benchmark.json}"
benchmark_plot="${HPML_BENCHMARK_PLOT_OUTPUT:-HPML_WORK/linux/results/insomnia-${job_id}-benchmark.png}"
training_output_dir="${HPML_TRAIN_OUTPUT_DIR:-HPML_WORK/linux/results/insomnia-${job_id}-training}"
training_metrics="${HPML_TRAIN_METRICS_OUTPUT:-HPML_WORK/linux/results/insomnia-${job_id}-training.json}"
model="${HPML_MODEL:-models/Qwen3-0.6B}"
dtype="${HPML_DTYPE:-bfloat16}"
memory_utilization="${HPML_VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
run_training="${HPML_RUN_TRAINING:-1}"
benchmark_label_args=()
if [[ -n "${HPML_BENCHMARK_LABEL:-}" ]]; then
    benchmark_label_args=(--label "$HPML_BENCHMARK_LABEL")
fi

python -c 'import torch, vllm; from trl.import_utils import is_vllm_available; assert torch.cuda.is_available(), "CUDA is not available"; assert is_vllm_available(), "vLLM failed to import"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__); print("vLLM:", vllm.__version__)'

python HPML_WORK/linux/scripts/issue7429_live_benchmark.py \
    --model "$model" \
    --turns "${HPML_TURNS:-2}" \
    --iterations "${HPML_BENCHMARK_ITERATIONS:-5}" \
    --dtype "$dtype" \
    --vllm-gpu-memory-utilization "$memory_utilization" \
    --output "$benchmark_output" \
    "${benchmark_label_args[@]}"

python HPML_WORK/shared/scripts/visualize_benchmark.py \
    --input "$benchmark_output" \
    --output "$benchmark_plot"

case "$run_training" in
    1|true|TRUE|yes|YES)
        python HPML_WORK/linux/scripts/train_grpo_benchmark.py \
            --model "$model" \
            --max-steps "${HPML_TRAIN_STEPS:-10}" \
            --dtype "$dtype" \
            --vllm-gpu-memory-utilization "$memory_utilization" \
            --output-dir "$training_output_dir" \
            --metrics-output "$training_metrics"
        python HPML_WORK/shared/scripts/visualize_benchmark.py \
            --input "$training_metrics" \
            --output "$training_output_dir/training-visualization.png"
        ;;
    0|false|FALSE|no|NO)
        echo "Skipping the GRPO training phase (HPML_RUN_TRAINING=$run_training)."
        ;;
    *)
        echo "HPML_RUN_TRAINING must be a boolean-like value; got: $run_training" >&2
        exit 1
        ;;
esac
