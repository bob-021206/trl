# Multi-turn tool-calling benchmark data

Datasets for measuring two optimizations of multi-turn tool calling in TRL's `GRPOTrainer` with
`vllm_mode="colocate"` + sleep mode:

- **Tech 1**: avoid reloading the weights after every tool-calling turn.
- **Tech 2**: keep vLLM's prefix cache (KV cache) across turns.

The gain of both depends on how many turns each sample has, how long each tool result is and how long the system
prompt is. So the number of tool calls per sample is **fixed by the data, not by the model**, and every length is
controlled exactly: differences in step time can then be attributed to the knobs.

## Datasets

| | W1 chain arithmetic | W2 multi-hop retrieval | W3 MuSiQue |
|---|---|---|---|
| Source | Generated | Generated fictional knowledge base (5000 entities) | `dgslibisey/MuSiQue`, validation split |
| Tool | `get_op(op_id)` | `search(query)` | `search(query)` (BM25 over the question's 20 paragraphs) |
| Turns K | Fixed: 2, 4, 8, 16 | Fixed: 2, 3, 4, 6 | Annotated hops 2–4; the number of searches is up to the model |
| Tokens per turn `L_obs` | Exactly 64 / 256 / 1024 | ≈ 266 / 773 / 1281 (1 / 3 / 5 documents) | Varies (136–1185) |
| System block `P` | Exactly 256 / 2048 | Exactly 512 | Exactly 512 |
| Cells × samples | 24 × 512 | 12 × 512 | 3 cells (512 / 512 / 332) |
| Role | **Main experiment**: sweep K, `L_obs`, `P` | Supplement: realistic long-context retrieval | Showcase: gains are not limited to synthetic data |

- **W1 – chain arithmetic.** Each step's op and the id of the next step are only in the previous tool result, so the
  model must call `get_op` exactly K times, one call per turn. The prompt asks for the running value after each result
  (`45 - 2 = 43`); the last result says it was the last step.
- **W2 – multi-hop retrieval.** The next entity is only in the document of the current one, so the model must call
  `search` exactly K times.
- **W3 – MuSiQue.** Each question carries its own 20 paragraphs. Since the number of searches and `L_obs` vary, report
  results by the measured number of calls. The answer reward is lenient: correct if a normalized gold answer or alias
  appears in the predicted answer.

Lengths are padded with English wikitext-2 text. Everything lives in this folder; generated data goes to `out/`
(~264 MB, git-ignored).

| File | Purpose |
|---|---|
| `common.py` | Token-length measurement under the Qwen3 chat template, wikitext filler corpus |
| `w*_env.py` | Environment, reward functions, reference trace (import these from the training script) |
| `w*_generate.py` | Dataset generators, one directory per grid cell |
| `w2_kb.py` | Builds the W2 knowledge base and BM25 index (once) |
| `w*_check.py` | Static checks, scripted-policy replay, regenerate-and-compare |
| `generate_all.sh` | Generates and checks all three workloads in one go |
| `smoke_test.py` | GRPO steps on each cell (GPU); Qwen3-0.6B by default, `--model`, `--lora-r`, `--max-steps` |

## Generate

One command, CPU only (add `--install` to first install the pinned `requirements.txt`):

```bash
bash HPML_WORK/data_prepare/generate_all.sh
```

Or step by step, from the repository root:

```bash
pip install -r HPML_WORK/data_prepare/requirements.txt
python HPML_WORK/data_prepare/w1_generate.py && python HPML_WORK/data_prepare/w1_check.py   # 24 cells, ~220 MB
python HPML_WORK/data_prepare/w2_kb.py                                                      # 5000 entities + BM25 index
python HPML_WORK/data_prepare/w2_generate.py && python HPML_WORK/data_prepare/w2_check.py   # 12 cells, ~25 MB
python HPML_WORK/data_prepare/w3_generate.py && python HPML_WORK/data_prepare/w3_check.py   # 3 cells, ~17 MB
```

`--preview` prints the first sample of a cell without saving, e.g.
`python HPML_WORK/data_prepare/w1_generate.py --K 3 --L-obs 64 --P 256 --num-samples 1 --preview`.
Everything is seeded (`--seed`, default 0); the token counts depend on the tokenizer version, hence the pinned
`requirements.txt`.

## Knobs

All lengths are Qwen3 tokens measured with `enable_thinking=False`, the way `GRPOTrainer` concatenates token ids.

| Knob | Meaning | W1 | W2 |
|---|---|---|---|
| `K` | tool calls per sample | 2, 4, 8, 16 | 2, 3, 4, 6 |
| `L_obs` | tokens appended per tool turn: the whole `GRPOTrainer._get_tool_suffix_ids` output, i.e. tool result + template + next generation prompt (about 34 of them are fixed overhead) | exactly 64, 256, 1024 | ≈ 266, 773, 1281 (`top_k` = 1, 3, 5; ±4 tokens) |
| `P` | rendered system block: system prompt + tool schema (the schema alone is ~190). Identical across the samples of a cell, like a real agent's system prompt | exactly 256, 2048 | exactly 512 |
| `L_doc` | tokens per W2 document | – | exactly 250 |

The user message comes after `P` and is ~100 tokens; `meta.json` records `max_prompt_len`.

## Use in training

Each cell directory loads with `datasets.load_from_disk` and has a `meta.json` with its `grpo_config` budgets
(`max_completion_length`, `vllm_max_model_length`, `max_tool_calling_iterations`; tables below) to unpack into
`GRPOConfig`. For example, for W1:

```python
import json, sys
from datasets import load_from_disk
sys.path.insert(0, "HPML_WORK/data_prepare")
from w1_env import ChainArithmeticEnv, answer_reward, steps_followed_reward

cell = "HPML_WORK/data_prepare/out/w1/K8_L256_P256"
budget = json.load(open(f"{cell}/meta.json"))["grpo_config"]
trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_from_disk(cell),
    environment_factory=ChainArithmeticEnv,
    reward_funcs=[answer_reward, steps_followed_reward],
    args=GRPOConfig(
        reward_weights=[1.0, 0.0],  # steps_followed is only logged: its mean is the share of the chain followed
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=True,  # without it the baseline has no per-turn reload
        chat_template_kwargs={"enable_thinking": False},
        **budget,
        ...
    ),
)
```

For W2, load the knowledge base once and share it:
`kb = KnowledgeBase("HPML_WORK/data_prepare/out/w2/kb")`, `environment_factory=lambda: KnowledgeSearchEnv(kb)`,
`reward_funcs=[answer_reward, hops_followed_reward]` from `w2_env`. For W3: `environment_factory=MusiqueSearchEnv`,
`reward_funcs=[answer_reward, supporting_found_reward]` from `w3_env`. [`smoke_test.py`](smoke_test.py) wires all
three.

`max_completion_length` is the longest reference trace in the cell plus 32 tokens per generated turn, rounded up to 64.

## Checks

The `*_check.py` scripts verify on CPU, for every sample:

- the answer, recomputed independently;
- each id / entity is reachable only from its predecessor (W1: every op_id appears only in the question or the
  previous result; W2: later hops are not named in the question, gold document in top-k at every hop);
- every length knob is exact (W1 tool suffixes and `P`; W2 documents and `P`);
- a scripted policy that reads only the tool results makes exactly K calls, follows every step and gets reward 1;
- the saved rows match `meta.json`, and regenerating from the seed gives the same rows and budgets (same sha256).

W3 keeps only the questions whose annotated sub-questions (with `#n` replaced by the previous answer) retrieve every
supporting paragraph in the top-3; its check replays those reference queries. All 39 cells pass.

## Model results

Needs a GPU environment with `vllm==0.32.0`, this repo's TRL (`pip install -e .`) and `bm25s` (setup in
[Hardware and software](#hardware-and-software)).

```bash
# Qwen3-0.6B, 1 step per cell, to check everything runs
python HPML_WORK/data_prepare/smoke_test.py HPML_WORK/data_prepare/out/w*/K*
# Qwen3-4B + LoRA, 50 steps
python HPML_WORK/data_prepare/smoke_test.py HPML_WORK/data_prepare/out/w1/K8_L256_P256 \
    --model Qwen/Qwen3-4B --lora-r 16 --learning-rate 1e-5 --max-steps 50
```

Per-cell numbers are in `out/smoke/*.json`.

### Does it run (Qwen3-0.6B, 1 GRPO step per cell, no warmup)

**All 39 cells train one step without errors**, with no tool-call failures and no completion clipped by the budget.
But the untrained 0.6B model does not call the tools as asked (16 rollouts per cell, so the rates are rough):

| Workload | Rollouts that follow the whole chain | Tool calls / sample | Answer reward |
|---|---|---|---|
| W1 | 0–38% per cell, mostly < 20% | 0–5: often answers without calling `get_op` | ~0 |
| W2 | 0–19%, 0% for K ≥ 4 | 0.9–2.4: mostly stops after the first search | ~0 |
| W3 | 0% | 0.06–0.3: answers from memory without searching | ~0 |

### Capability by model size (inference only, greedy, 32 samples per cell, same tool loop)

| Model | W1 K4_L256_P256 acc / followed | W1 K8_L256_P256 acc / followed | W2 K3_top3 acc | W3 K2_top3 acc |
|---|---|---|---|---|
| Qwen3-0.6B | 0% / 0% (copies the prompt example instead of calling) | 0% / 0% | 0% | 0% |
| Qwen3-1.7B | 0% / 94% (follows, but never writes the running value) | 0% / 94% | 6% | 0% |
| Qwen3-4B | 47% / 100% | 59% / 97% | 47% | 19% |

The 4B traces write each running value and end on the stored answer, so the labels are right: the low accuracy of
0.6B and 1.7B is model size, not the data.

### 50 GRPO steps (W1 K8_L256_P256, 16 rollouts per step)

| | Qwen3-0.6B, full fine-tuning | Qwen3-4B + LoRA (rank 16) |
|---|---|---|
| Options | `--followed-weight 1.0 --learning-rate 3e-6` | `--lora-r 16 --learning-rate 1e-5` |
| Reward | Answer + share of the chain followed (1:1) | Answer only |
| Share of the chain followed | 0 at step 1 → 0.53 at step 7 → 0.88 at step 10 → 1.0 at step 15; 8 of the last 10 steps at 1.0 (others 0.98, 0.77) | 1.0 on 45 of 50 steps, lowest 0.78 |
| Tool calls per sample | 7.8–8.3 over the last 10 steps | 8.0–8.3 (two steps at 6.6 and 7.6) |
| Answer reward | ~0 (non-zero on 2 of 50 steps) | Mean 0.26 over steps 1–10 → 0.54 over 11–20 → 0.88 over 21–30; 1.0 on 15 of the last 20 (0.69–0.94 on the others) |
| Step time | 27–37 s (step 1, without tool calls, 6 s); 1726 s total | 69–91 s; 4208 s total |

GRPO alone fixes the turn count, no SFT needed. 4B result: `out/smoke/w1_K8_L256_P256_4b_lora.json`.

**Longest cell.** 4B + LoRA on W1 K16_L1024_P2048 ran 2 steps without OOM and with nothing clipped: 7/16 rollouts
followed all 16 turns (~19k-token sequences). Each step takes ~1450 s (24 min), and the cell is hard for 4B (share of
the chain followed 77% and 47%, accuracy 0 and 6%).

## Recommendations

**Use Qwen3-4B + LoRA for the actual experiments, and Qwen3-0.6B only to debug the pipeline.**

- **4B is the only model that does both**: a stable number of turns (8 tool calls per sample), so Tech 1 and Tech 2 can
  be timed accurately, and accuracy that goes from ~26% to ~100%, which shows the optimizations don't change what
  training learns.
- **LoRA doesn't get in the way of measuring Tech 1.** TRL merges the adapter and syncs the full 4B weights to vLLM, so
  the per-turn weight reload is still there, and grows with model size.
- **4B cannot be fully fine-tuned on 48 GB**: bf16 weights + gradients + fp32 Adam states alone are ~48 GB, before
  activations and vLLM. 8B and larger fit even less.
- **0.6B does not learn the task**: after training it runs all K turns but accuracy stays ~0, so it only serves to
  validate scripts and get rough timings. 1.7B follows the chain but doesn't write the running values; accuracy is 0
  too.

Use of the datasets:

- **W1 for the main experiment.** Fix `P`=256, sweep K and `L_obs`, and compare the step time with and without the
  optimizations. A few steps per cell are enough.
- **2–3 W2 cells as a supplement**, e.g. K3_top3 and K6_top3.
- **A single W3 cell as a showcase**, with results grouped by the measured number of calls.

Open risks:

- **The longest cells are slow** (~24 min per step on K16_L1024_P2048): time a few steps there, don't train to
  convergence.
- **Accuracy saturates easily.** Near the end of 50 steps on K8 every group is all-correct, the advantages vanish and
  the curves of two runs overlap. To compare accuracy between runs, use a harder cell (larger K) or fewer steps.

## Hardware and software

| Item | Configuration |
|---|---|
| GPU | 1 × NVIDIA RTX A6000, 48 GB (about 45 GiB usable) |
| Driver / CUDA | 595.91.07 / 13.2 |
| CPU | Intel Xeon Gold 6342 @ 2.80GHz, 96 threads |
| RAM | 503 GB |
| Training environment | Python 3.12.3, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, peft 0.21.2, TRL 1.17.0.dev0 (this repo) |
| Data generation environment (CPU only) | transformers 5.22.2, tokenizers 0.23.3, datasets 5.1.0, bm25s 0.3.16 (`requirements.txt`) |
| Training settings | vLLM colocate + sleep mode, `vllm_gpu_memory_utilization=0.5`, bf16, gradient checkpointing, 4 prompts × 4 generations per step |

- vLLM and training share the GPU. `nvidia-smi` shows it almost full (~45 GB) while running: that is vLLM reserving
  its KV cache, not a sign of running out of memory.
- `peak_memory_gb` (`torch.cuda.max_memory_allocated`) is unreliable under sleep mode: it reports 61.9 GB on the
  longest cell, more than the GPU has, likely because the memory vLLM frees and reallocates every turn is counted
  repeatedly. Check for OOM instead.
- Every process prints `CUDA Error: invalid argument at cumem_allocator.cpp` at exit: vLLM sleep-mode teardown, after
  training, exit code 0, no effect on the results.
- The repo's environment setup script requires CUDA 12.9 or 13.0; this machine's driver is 13.2, so the GPU
  environment was installed separately.

## Budgets per cell

W1 (512 samples each):

| K | L_obs | P | longest reference completion | `max_completion_length` | `vllm_max_model_length` | `max_tool_calling_iterations` |
|---|---|---|---|---|---|---|
| 2 | 64 | 256 | 216 | 320 | 768 | 4 |
| 2 | 64 | 2048 | 216 | 320 | 2560 | 4 |
| 2 | 256 | 256 | 599 | 704 | 1152 | 4 |
| 2 | 256 | 2048 | 600 | 704 | 2944 | 4 |
| 2 | 1024 | 256 | 2136 | 2240 | 2688 | 4 |
| 2 | 1024 | 2048 | 2135 | 2240 | 4480 | 4 |
| 4 | 64 | 256 | 414 | 576 | 1024 | 6 |
| 4 | 64 | 2048 | 414 | 576 | 2816 | 6 |
| 4 | 256 | 256 | 1182 | 1344 | 1792 | 6 |
| 4 | 256 | 2048 | 1183 | 1344 | 3584 | 6 |
| 4 | 1024 | 256 | 4254 | 4416 | 4864 | 6 |
| 4 | 1024 | 2048 | 4254 | 4416 | 6656 | 6 |
| 8 | 64 | 256 | 811 | 1152 | 1600 | 10 |
| 8 | 64 | 2048 | 813 | 1152 | 3392 | 10 |
| 8 | 256 | 256 | 2347 | 2688 | 3136 | 10 |
| 8 | 256 | 2048 | 2348 | 2688 | 4928 | 10 |
| 8 | 1024 | 256 | 8491 | 8832 | 9280 | 10 |
| 8 | 1024 | 2048 | 8490 | 8832 | 11072 | 10 |
| 16 | 64 | 256 | 1605 | 2176 | 2624 | 18 |
| 16 | 64 | 2048 | 1606 | 2176 | 4416 | 18 |
| 16 | 256 | 256 | 4679 | 5248 | 5696 | 18 |
| 16 | 256 | 2048 | 4679 | 5248 | 7488 | 18 |
| 16 | 1024 | 256 | 16965 | 17536 | 17984 | 18 |
| 16 | 1024 | 2048 | 16965 | 17536 | 19776 | 18 |

W2 (512 samples each, `P` = 512, `L_doc` = 250):

| K | top_k | mean L_obs | longest reference completion | `max_completion_length` | `vllm_max_model_length` | `max_tool_calling_iterations` |
|---|---|---|---|---|---|---|
| 2 | 1 | 265.7 | 648 | 768 | 1408 | 4 |
| 2 | 3 | 773.1 | 1661 | 1792 | 2432 | 4 |
| 2 | 5 | 1280.6 | 2677 | 2816 | 3456 | 4 |
| 3 | 1 | 265.7 | 955 | 1088 | 1728 | 5 |
| 3 | 3 | 773.1 | 2481 | 2624 | 3264 | 5 |
| 3 | 5 | 1280.6 | 4004 | 4160 | 4800 | 5 |
| 4 | 1 | 265.7 | 1273 | 1472 | 2112 | 6 |
| 4 | 3 | 773.0 | 3295 | 3456 | 4096 | 6 |
| 4 | 5 | 1280.6 | 5327 | 5504 | 6144 | 6 |
| 6 | 1 | 265.7 | 1883 | 2112 | 2752 | 8 |
| 6 | 3 | 773.1 | 4936 | 5184 | 5824 | 8 |
| 6 | 5 | 1280.5 | 7974 | 8256 | 8896 | 8 |

W3 (`top_k` = 3, `P` = 512; `max_completion_length` budgets K + 2 searches that each return the 3 longest
paragraphs of the question):

| K | kept / candidates | L_obs of the reference trace | longest reference completion | `max_completion_length` | `vllm_max_model_length` | `max_tool_calling_iterations` |
|---|---|---|---|---|---|---|
| 2 | 512/1252 | 156–1185 (mean 383.0) | 1980 | 7360 | 8000 | 4 |
| 3 | 512/760 | 139–1087 (mean 421.4) | 2584 | 7296 | 7936 | 5 |
| 4 | 332/405 | 136–906 (mean 391.2) | 2798 | 10048 | 10688 | 6 |

## Notes for the analysis

- Weight syncs per step under the baseline = 1 + the largest number of turns in the batch (one generate call per turn
  for the whole batch), not K per sample.
- vLLM's prefix cache already works *within* one generate call, so even the baseline prefills the shared system prompt
  once per turn per batch, not once per sample. In W1 the G rollouts of a prompt receive identical tool results and
  near-identical outputs, so their histories may coincide token for token; report the number of distinct prefixes in
  the batch next to the measured prefill tokens, otherwise the Tech 2 gain is underestimated relative to
  `K·P + K²/2·(L_gen + L_obs)`.
- Prompts are in English (the filler corpus is English wikitext-2). The `confusable` filler mode of the proposal is not
  implemented.
