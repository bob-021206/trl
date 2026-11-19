"""Reuse Yidan's workloads, preserving tool schemas and reward definitions."""

import copy
import hashlib
import inspect
import json
import random
import sys
from functools import wraps
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data_prepare"


def sample_id(row):
    # Group duplicate questions together, even if metadata/row indices differ.
    return hashlib.sha256(json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_indices(rows, eval_samples, split_seed, train_samples=0):
    """Deterministic prompt-disjoint split; eval_samples counts unique prompts."""
    ids = [sample_id(row) for row in rows]
    unique = sorted(set(ids))
    random.Random(split_seed).shuffle(unique)
    if eval_samples >= len(unique):
        raise ValueError("Holdout must leave at least one distinct training prompt")
    heldout = set(unique[:eval_samples])
    seen = set()
    train, evaluation = [], []
    for index, key in enumerate(ids):
        if key in heldout:
            if key not in seen:
                evaluation.append(index)
                seen.add(key)
        else:
            train.append(index)
    if train_samples:
        if train_samples > len(train):
            raise ValueError("train_samples exceeds available training rows")
        train = train[:train_samples]
    return train, evaluation, ids


def measured_workload(factory, rewards, followed_all, recorder, demo_samples):
    """Wrap existing environment methods without adding tools or changing their signatures."""
    classes = {}
    episode = 0
    demos = []

    def measured_factory():
        instance = factory()
        base = type(instance)
        if base not in classes:
            original_reset = base.reset

            @wraps(original_reset)
            def reset(self, *args, **kwargs):
                nonlocal episode
                self._ann_episode = episode
                episode += 1
                self._ann_sample = kwargs.get("ann_sample_id")
                self._ann_calls = 0
                self._ann_errors = 0
                with recorder.span("environment_reset"):
                    return original_reset(self, *args, **kwargs)

            methods = {"reset": reset}
            for name, original in inspect.getmembers(base, predicate=inspect.isfunction):
                if name.startswith("_") or name in ("reset", "get_reward"):
                    continue
                if inspect.iscoroutinefunction(original):
                    raise ValueError("This collector supports Yidan's synchronous CPU tools only")

                def wrap_tool(method):
                    @wraps(method)
                    def call(self, *args, **kwargs):
                        self._ann_calls += 1
                        with recorder.span(
                            "cpu_tool", tool=method.__name__, episode=self._ann_episode, sample_id=self._ann_sample
                        ) as fields:
                            try:
                                result = method(self, *args, **kwargs)
                            except Exception:
                                self._ann_errors += 1
                                raise
                            failed = isinstance(result, str) and result.startswith("error:")
                            self._ann_errors += int(failed)
                            fields["tool_error"] = failed
                            return result

                    return call

                methods[name] = wrap_tool(original)
            classes[base] = type(f"Measured{base.__name__}", (base,), methods)
        # These are ordinary Python classes. Preserve the factory's initialized/shared KB.
        instance.__class__ = classes[base]
        return instance

    original_reward = rewards[0]

    @wraps(original_reward)
    def measured_reward(completions, environments, **kwargs):
        values = original_reward(completions=completions, environments=environments, **kwargs)
        for index, (value, env, completion) in enumerate(zip(values, environments, completions, strict=True)):
            fields = dict(
                sample_id=env._ann_sample,
                episode=env._ann_episode,
                correct=value,
                tool_calls=env._ann_calls,
                tool_errors=env._ann_errors,
                intended_k=kwargs.get("K", [None] * len(values))[index],
                fully_followed=bool(followed_all(env)),
                assistant_messages=sum(m["role"] == "assistant" for m in completion),
            )
            with recorder.span("quality_sample", **fields):
                pass
            # Keep bounded real interactions for the demo report; no full-token logging in hot loops.
            if sum(d["phase"] == recorder.phase for d in demos) < demo_samples:
                demos.append(
                    copy.deepcopy(
                        {
                            **fields,
                            "phase": recorder.phase,
                            "step": recorder.step,
                            "rollout": recorder.rollout,
                            "prompt": kwargs.get("prompts", [None] * len(values))[index],
                            "completion": completion,
                        }
                    )
                )
        return values

    return measured_factory, [measured_reward, *rewards[1:]], demos


def load_workload(cell, config, recorder):
    # Yidan's module uses sibling imports. Keep the directory visible for runtime imports too.
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    from common import rows_sha256
    from datasets import load_from_disk
    from HPML_WORK.data_prepare.smoke_test import workload

    meta = json.loads((cell / "meta.json").read_text())
    data = load_from_disk(str(cell))
    rows = data.to_list()
    if rows_sha256(rows) != meta["sha256"]:
        raise ValueError("Dataset content does not match meta.json sha256")
    train_indices, eval_indices, ids = split_indices(
        rows, config["eval_samples"], config["split_seed"], config["train_samples"]
    )
    data = data.add_column("ann_sample_id", ids)
    if (
        len(train_indices)
        < config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] // config["num_generations"]
    ):
        raise ValueError("Not enough training prompts for one generation batch")
    factory, rewards, full = workload(cell)
    if cell.parent.name == "w2":
        kb_meta = json.loads((cell.parent / "kb/meta.json").read_text())
        if kb_meta["sha256"] != meta["kb_sha256"]:
            raise ValueError("W2 cell refers to a different knowledge base")
    factory, rewards, demos = measured_workload(factory, rewards, full, recorder, config["demo_samples"])
    split = {
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_prompt_ids": [ids[i] for i in train_indices],
        "eval_prompt_ids": [ids[i] for i in eval_indices],
        "split_seed": config["split_seed"],
    }
    return data.select(train_indices), data.select(eval_indices), factory, rewards, meta, split, demos
