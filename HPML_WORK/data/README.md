# Multi-turn tool-calling benchmark data

Datasets for measuring two colocate/sleep-mode optimizations of multi-turn tool calling in `GRPOTrainer`: avoiding the
per-turn weight reload (Tech 1) and keeping the prefix cache across turns (Tech 2). The number of tool calls per
sample is fixed by the data, not by the model, so the cost of each optimization can be attributed to the knobs.

- **W1 – chain arithmetic** (main grid). Each step's op and the id of the next step are only in the previous tool
  result, so the model must call `get_op` exactly K times, one call per turn. The prompt asks for the running value
  after each result (`45 - 2 = 43`); the last result says it was the last step.
- **W2 – multi-hop retrieval** (realistic long-context workload). Fictional knowledge base; the next entity is only in
  the document of the current one, so the model must call `search` exactly K times.
- **W3 – MuSiQue** (public multi-hop QA, `dgslibisey/MuSiQue` validation split). Each question carries its own 20
  paragraphs and `search` runs BM25 over them. K is the annotated hop count (2–4), but the number of searches is
  up to the model, and `L_obs` varies with the paragraphs: report results by the measured number of calls. The
  answer reward is lenient: correct if a normalized gold answer or alias appears in the predicted answer.

Everything lives in this folder; generated data goes to `out/` (git-ignored).

| File | Purpose |
|---|---|
| `common.py` | Token-length measurement under the Qwen3 chat template, wikitext filler corpus |
| `w*_env.py` | Environment, reward functions, reference trace (import these from the training script) |
| `w*_generate.py` | Dataset generators, one directory per grid cell |
| `w2_kb.py` | Builds the W2 knowledge base and BM25 index (once) |
| `w*_check.py` | Static checks, scripted-policy replay, regenerate-and-compare |
| `generate_all.sh` | Generates and checks all three workloads in one go |
| `smoke_test.py` | GRPO steps on each cell (GPU); Qwen3-0.6B by default, `--model`, `--lora-r`, `--max-steps` |
| `REPORT.md` | Write-up (Chinese): data design, usage, model results, hardware, recommended model |

## Generate

One command (add `--install` to first install `requirements.txt`):

```bash
bash HPML_WORK/data/generate_all.sh
```

Or step by step, from the repository root:

```bash
pip install -r HPML_WORK/data/requirements.txt
python HPML_WORK/data/w1_generate.py && python HPML_WORK/data/w1_check.py   # 24 cells, ~220 MB
python HPML_WORK/data/w2_kb.py                                              # 5000 entities + BM25 index
python HPML_WORK/data/w2_generate.py && python HPML_WORK/data/w2_check.py   # 12 cells, ~25 MB
python HPML_WORK/data/w3_generate.py && python HPML_WORK/data/w3_check.py   # 3 cells, ~17 MB
```

`--preview` prints the first sample of a cell without saving, e.g.
`python HPML_WORK/data/w1_generate.py --K 3 --L-obs 64 --P 256 --num-samples 1 --preview`.
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

Each cell directory has a `meta.json` with its `grpo_config` budgets (tables below). For example, for W1:

```python
import json, sys
from datasets import load_from_disk
sys.path.insert(0, "HPML_WORK/data")
from w1_env import ChainArithmeticEnv, answer_reward, steps_followed_reward

cell = "HPML_WORK/data/out/w1/K8_L256_P256"
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
`kb = KnowledgeBase("HPML_WORK/data/out/w2/kb")`, `environment_factory=lambda: KnowledgeSearchEnv(kb)`,
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
- the saved rows match `meta.json`, and regenerating from the seed gives the same rows and budgets.

W3 keeps only the questions whose annotated sub-questions (with `#n` replaced by the previous answer) retrieve every
supporting paragraph in the top-3; its check replays those reference queries.

## Smoke test with TRL (Qwen3-0.6B, no warmup)

```bash
python HPML_WORK/data/smoke_test.py HPML_WORK/data/out/w*/K*
```

Needs a GPU environment with `vllm==0.32.0`, this repo's TRL (`pip install -e .`) and `bm25s`. One GRPO step per
cell (vLLM colocate + sleep mode, 4 prompts x 4 generations, 1×A6000, vLLM 0.32.0). **All 39
cells train one step without errors**, with no tool-call failures and no completion clipped by the budget.
Each process prints `CUDA Error: invalid argument at cumem_allocator.cpp` at exit
(vLLM sleep-mode teardown, after training; exit code 0).

The base 0.6B model is not yet usable for the benchmark: the share of rollouts that follow the whole chain is far
below the 95% gate. GRPO with the chain-following reward fixes that (below).

| Workload | Rollouts that follow the whole chain | Tool calls / sample | Answer reward |
|---|---|---|---|
| W1 | 0–38% per cell, mostly < 20% | 0–5: often answers without calling `get_op` | ~0 |
| W2 | 0–19%, 0% for K ≥ 4 | 0.9–2.4: mostly stops after the first search | ~0 |
| W3 | 0% | 0.06–0.3: answers from memory without searching | ~0 |

16 rollouts per cell, so the rates are rough. Per-cell numbers are in `out/smoke/*.json`.

**GRPO alone fixes the turn count, no SFT needed.** With the chain-following metric in the reward
(`--followed-weight 1.0 --learning-rate 3e-6 --max-steps 50`, W1 K8_L256_P256), the share of the chain followed
goes from 0 at step 1 to 0.53 at step 7, 0.88 at step 10 and 1.0 at step 15; in the last 10 steps, 8 have every
rollout following the whole chain (the other two: 0.98 and 0.77), with 7.8–8.3 tool calls per sample. The answer
reward stays ~0 (non-zero on 2 of 50 steps). One step takes 27–37 s (16 rollouts × 8 turns; step 1, without tool calls, 6 s), 1726 s in total.

**Low accuracy is model size, not the data.** Inference only (greedy, 32 samples per cell, same tool loop):

| Model | W1 K4_L256_P256 acc / followed | W1 K8_L256_P256 acc / followed | W2 K3_top3 acc | W3 K2_top3 acc |
|---|---|---|---|---|
| Qwen3-0.6B | 0% / 0% (copies the prompt example instead of calling) | 0% / 0% | 0% | 0% |
| Qwen3-1.7B | 0% / 94% (follows, but never writes the running value) | 0% / 94% | 6% | 0% |
| Qwen3-4B | 47% / 100% | 59% / 97% | 47% | 19% |

The 4B traces write each running value and end on the stored answer, so the labels are right.

**4B with LoRA learns the task.** `--model Qwen/Qwen3-4B --lora-r 16 --learning-rate 1e-5 --max-steps 50` on W1
K8_L256_P256 (answer reward only; full fine-tuning does not fit a 48 GB GPU): the answer reward averages 0.26 over
steps 1–10, 0.54 over 11–20, 0.88 over 21–30 and is 1.0 on 15 of the last 20 steps (0.69–0.94 on the others).
Tool calls stay at 8.0–8.3 per sample (two steps at 6.6 and 7.6). One step takes 69–91 s, 4208 s in total, peak memory
40.6 GB. Near the end every group is all-correct, so the advantages vanish: to compare accuracy curves between
runs, use a harder cell (larger K) or fewer steps. Result: `out/smoke/w1_K8_L256_P256_4b_lora.json`.

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
