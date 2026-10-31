#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_DEFAULT_TIMEOUT=60

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found. Run setup on a GPU compute node with NVIDIA drivers available." >&2
    exit 1
fi

if ! command -v "${CC:-cc}" >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y build-essential
    else
        echo "A C compiler is required by Triton but '${CC:-cc}' was not found." >&2
        echo "Install build-essential on Ubuntu, load a compiler module, or set CC to an installed compiler." >&2
        exit 1
    fi
fi

python_cmd="${PYTHON:-}"
if [ -z "$python_cmd" ]; then
    for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            python_cmd="$candidate"
            break
        fi
    done
fi

if [ -z "$python_cmd" ] && command -v module >/dev/null 2>&1; then
    echo "No Python 3.11+ found on PATH; loading the Insomnia Anaconda module."
    module load anaconda/2023.09
    for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && \
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            python_cmd="$candidate"
            break
        fi
    done
fi

if [ -z "$python_cmd" ] || ! command -v "$python_cmd" >/dev/null 2>&1; then
    echo "Python 3.11 or newer is required. Load a compatible Python module or set PYTHON=/path/to/python3.11." >&2
    exit 1
fi
if ! "$python_cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "Python 3.11 or newer is required; '$python_cmd' is too old." >&2
    exit 1
fi

python_version="$("$python_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_include="$("$python_cmd" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [ ! -f "$python_include/Python.h" ]; then
    if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y "python${python_version}-dev"
    else
        echo "Python development headers are required by Triton, but '$python_include/Python.h' was not found." >&2
        echo "Install python${python_version}-dev on Ubuntu, or load a Python module that includes its development headers." >&2
        exit 1
    fi
fi

cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
case "$cuda_version" in
    12.9)
        torch_index="https://download.pytorch.org/whl/cu129"
        torchcodec_version="0.16.0+cu129"
        vllm_wheel="https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        ;;
    13.0)
        torch_index="https://download.pytorch.org/whl/cu130"
        torchcodec_version="0.16.0+cu130"
        ;;
    *)
        echo "Unsupported or undetected CUDA driver version: '${cuda_version:-unknown}'. Supported versions are 12.9 and 13.0." >&2
        exit 1
        ;;
esac

"$python_cmd" -m venv --clear .venv-issue7429
source .venv-issue7429/bin/activate
python -m pip install --upgrade pip
if [ "$cuda_version" = "12.9" ]; then
    echo "Installing PyTorch and vLLM for CUDA 12.9."
    python -m pip install "torch==2.13.0" --index-url "$torch_index"
    python -m pip install "$vllm_wheel" --extra-index-url "$torch_index"
else
    echo "Installing PyTorch and vLLM for CUDA 13.0."
    python -m pip install "vllm==0.32.0"
fi
python -m pip install -r HPML_WORK/linux/requirements.txt
python -m pip install --force-reinstall --no-deps \
    --index-url "$torch_index" "torchcodec==$torchcodec_version"

EXPECTED_CUDA_VERSION="$cuda_version" python -c 'import os, torch, torchcodec, vllm; from trl.import_utils import is_vllm_available; expected_cuda = os.environ["EXPECTED_CUDA_VERSION"]; assert torch.version.cuda and torch.version.cuda.startswith(expected_cuda), f"PyTorch CUDA {torch.version.cuda} does not match driver CUDA {expected_cuda}"; assert torch.cuda.is_available(), "CUDA GPU is not visible to PyTorch"; assert is_vllm_available(), "vLLM failed to import"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__); print("PyTorch CUDA:", torch.version.cuda); print("TorchCodec:", torchcodec.__version__); print("vLLM:", vllm.__version__)'
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
python HPML_WORK/shared/scripts/download_qwen3.py --local-dir models/Qwen3-0.6B
