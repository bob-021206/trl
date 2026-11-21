"""CPU contract tests against Yidan's environments; no model download or synthetic GPU results."""

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

from HPML_WORK.data_prepare.w1_env import ChainArithmeticEnv, answer_reward, steps_followed_reward
from HPML_WORK.profiling.recorder import Recorder
from HPML_WORK.profiling.workloads import DATA_DIR, load_workload, measured_workload, split_indices


HAS_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "datasets", "peft", "bm25s"))


class WorkloadTests(unittest.TestCase):
    def test_duplicate_prompts_never_cross_split(self):
        rows = [{"prompt": [{"role": "user", "content": str(i // 2)}], "extra": i} for i in range(20)]
        train, evaluation, ids = split_indices(rows, 3, 1729)
        self.assertEqual(len(evaluation), 3)
        self.assertEqual(len(train), 14)
        self.assertTrue({ids[i] for i in train}.isdisjoint(ids[i] for i in evaluation))
        self.assertEqual((train, evaluation, ids), split_indices(rows, 3, 1729))
        with self.assertRaises(ValueError):
            split_indices(rows, 10, 1729)

    def test_w1_tools_and_rewards_preserve_behavior(self):
        rec = Recorder("unused")
        factory, rewards, demos = measured_workload(
            ChainArithmeticEnv,
            [answer_reward, steps_followed_reward],
            lambda env: env.steps_followed == len(env.op_ids),
            rec,
            1,
        )
        original, measured = ChainArithmeticEnv(), factory()
        row = dict(op_ids=["a", "b"], observations=["+2; next b", "+4; done"], ann_sample_id="question")
        for env in (original, measured):
            env.reset(**row)
        self.assertEqual(inspect.signature(original.get_op), inspect.signature(measured.get_op))
        self.assertEqual(original.get_op.__doc__, measured.get_op.__doc__)
        self.assertEqual(original.get_op("a"), measured.get_op("a"))
        self.assertEqual(original.get_op("b"), measured.get_op("b"))
        self.assertEqual(original.get_op("invalid"), measured.get_op("invalid"))
        completion = [[{"role": "assistant", "content": "<answer>7</answer>"}]]
        kwargs = dict(completions=completion, answer=[7], K=[2], prompts=[[{"role": "user", "content": "chain"}]])
        self.assertEqual(answer_reward(**kwargs), rewards[0](environments=[measured], **kwargs))
        self.assertIs(rewards[1], steps_followed_reward)
        quality = next(e["fields"] for e in rec.events if e["name"] == "quality_sample")
        self.assertEqual((quality["tool_calls"], quality["tool_errors"], quality["correct"]), (3, 1, 1.0))
        self.assertTrue(quality["fully_followed"])
        self.assertEqual(len(demos), 1)
        self.assertEqual(demos[0]["prompt"], kwargs["prompts"][0])

    @unittest.skipUnless(HAS_STACK, "Requires the team's Python data/training dependencies")
    def test_real_factory_w2_w3_and_transformers_tool_schema(self):
        import bm25s
        from transformers.utils import get_json_schema

        sys.path.insert(0, str(DATA_DIR))
        from HPML_WORK.data_prepare.smoke_test import workload

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kb = root / "w2/kb"
            kb.mkdir(parents=True)
            entities = [
                {"name": "Alpha", "doc": "Alpha points to Beta"},
                {"name": "Beta", "doc": "Beta has color blue"},
            ]
            (kb / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in entities))
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize([e["doc"] for e in entities], show_progress=False), show_progress=False)
            retriever.save(kb / "bm25")
            cases = [
                ("w1", dict(op_ids=["a"], observations=["+2"], K=1, answer=7), "get_op", ["a"], "7"),
                (
                    "w2",
                    dict(gold_entities=["Alpha", "Beta"], top_k=1, K=2, answer="blue"),
                    "search",
                    ["Alpha", "Beta"],
                    "blue",
                ),
                (
                    "w3",
                    dict(
                        docs=["Orion was born in Paris", "Paris is in France"],
                        supporting_idx=[0, 1],
                        top_k=2,
                        K=2,
                        answer="Paris",
                        answer_aliases=["City of Paris"],
                    ),
                    "search",
                    ["Orion", "France"],
                    "the City of Paris",
                ),
            ]
            for name, row, tool, queries, answer in cases:
                with self.subTest(workload=name):
                    rec = Recorder("unused")
                    original_factory, original_rewards, followed = workload(root / name / "cell")
                    factory, rewards, _ = measured_workload(original_factory, original_rewards, followed, rec, 0)
                    original, measured = original_factory(), factory()
                    self.assertEqual(
                        get_json_schema(getattr(original, tool)), get_json_schema(getattr(measured, tool))
                    )
                    if name == "w2":
                        self.assertIs(original.kb, measured.kb)
                    for env in (original, measured):
                        env.reset(**row, ann_sample_id=name)
                    for query in queries:
                        self.assertEqual(getattr(original, tool)(query), getattr(measured, tool)(query))
                    completion = [[{"role": "assistant", "content": f"<answer>{answer}</answer>"}]]
                    kwargs = {k: [v] for k, v in row.items()}
                    expected = original_rewards[0](completions=completion, environments=[original], **kwargs)
                    self.assertEqual(expected, rewards[0](completions=completion, environments=[measured], **kwargs))
                    self.assertTrue(followed(measured))
                    self.assertEqual(expected, [1.0])

    @unittest.skipUnless(HAS_STACK, "Requires the team's Python data/training dependencies")
    def test_load_real_dataset_format_and_reject_tampering(self):
        from datasets import Dataset

        sys.path.insert(0, str(DATA_DIR))
        from common import rows_sha256

        with tempfile.TemporaryDirectory() as temp:
            cell = Path(temp) / "w1/cell"
            rows = [
                dict(
                    prompt=[{"role": "user", "content": f"Question {i}"}],
                    op_ids=[str(i)],
                    observations=["+2"],
                    K=1,
                    answer=i + 2,
                )
                for i in range(10)
            ]
            Dataset.from_list(rows).save_to_disk(str(cell))
            meta = dict(sha256=rows_sha256(rows), grpo_config={"max_completion_length": 128})
            (cell / "meta.json").write_text(json.dumps(meta))
            config = dict(
                eval_samples=2,
                split_seed=1729,
                train_samples=0,
                demo_samples=0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                num_generations=2,
            )
            train, evaluation, factory, _, _, split, _ = load_workload(cell, config, Recorder("unused"))
            self.assertEqual((len(train), len(evaluation)), (8, 2))
            self.assertTrue(set(split["train_prompt_ids"]).isdisjoint(split["eval_prompt_ids"]))
            env = factory()
            env.reset(**train[0])
            self.assertEqual(env._ann_sample, train[0]["ann_sample_id"])
            meta["sha256"] = "tampered"
            (cell / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_workload(cell, config, Recorder("unused"))


if __name__ == "__main__":
    unittest.main()
