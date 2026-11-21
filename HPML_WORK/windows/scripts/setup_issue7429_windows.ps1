$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install an NVIDIA driver and confirm GPU access first."
}
nvidia-smi

$VenvPython = Join-Path $RepoRoot ".venv-issue7429\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    python -m venv .venv-issue7429
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $VenvPython -m pip install -r HPML_WORK\windows\requirements.txt

& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available in PyTorch'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__)"
& $VenvPython HPML_WORK\shared\scripts\download_qwen3.py --local-dir models\Qwen3-0.6B
