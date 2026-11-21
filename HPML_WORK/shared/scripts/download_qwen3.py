import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main():
    parser = argparse.ArgumentParser(description="Download the proposal's pinned Qwen3-0.6B checkpoint.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models") / "Qwen3-0.6B",
        help="Destination (defaults to models/Qwen3-0.6B at the repository root).",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    local_dir = args.local_dir if args.local_dir.is_absolute() else repo_root / args.local_dir
    model_dir = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=local_dir,
    )
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise RuntimeError(f"Expected a Qwen3 checkpoint, found model_type={config.get('model_type')!r}.")
    weight_files = list(Path(model_dir).glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}.")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    resolved_dir = Path(model_dir).resolve()
    display_dir = resolved_dir.relative_to(repo_root).as_posix() if resolved_dir.is_relative_to(repo_root) else str(resolved_dir)
    print(f"Downloaded to: {display_dir}")
    print(f"Weight files: {len(weight_files)}")


if __name__ == "__main__":
    main()
