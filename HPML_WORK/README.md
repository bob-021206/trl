# HPML project workspace

Project proposal and role assignments are in [`shared/docs`](./shared/docs/). Platform-specific instructions and scripts are intentionally separated:

- [Windows GPU scripts and guide](./windows/docs/ISSUE_7429_BENCHMARK.md)
- [Linux CUDA/vLLM scripts and guide](./linux/docs/ISSUE_7429_BENCHMARK.md)
- Shared model download and benchmark visualization scripts are in [`shared/scripts`](./shared/scripts/).
- The Insomnia Slurm training/benchmark job is [`insomnia_grpo_benchmark.sbatch`](./linux/scripts/insomnia_grpo_benchmark.sbatch).

Run commands from the repository root. Model files default to `models/Qwen3-0.6B`, and benchmark outputs are saved under the matching platform's `results` folder.
