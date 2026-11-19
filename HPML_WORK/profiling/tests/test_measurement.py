import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from HPML_WORK.profiling.instrumentation import instrument, validate_lifecycle
from HPML_WORK.profiling.recorder import Recorder
from HPML_WORK.profiling.run import ROOT, validate_config
from HPML_WORK.profiling.smoke import FakeTrainer, create_fixture
from HPML_WORK.profiling.summarize import reduce_events, summarize_run


class MeasurementTests(unittest.TestCase):
    def test_nested_exclusive_times_do_not_double_count(self):
        rec = Recorder("unused")
        with rec.span("outer"):
            with rec.span("inner"):
                pass
        inner, outer = rec.events
        self.assertEqual(inner["parent_id"], outer["event_id"])
        self.assertAlmostEqual(outer["exclusive_ms"] + inner["duration_ms"], outer["duration_ms"])

    def test_error_spans_close_and_hooks_restore(self):
        rec = Recorder("unused")
        trainer = FakeTrainer(rec)
        original = trainer.vllm_generation.generate
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with instrument(trainer, rec):
                with rec.span("failure"):
                    raise RuntimeError("failure")
        self.assertFalse(rec.stack)
        self.assertEqual(rec.events[-1]["status"], "error")
        self.assertEqual(trainer.vllm_generation.generate, original)
        self.assertNotIn("generate", vars(trainer.vllm_generation))

    def test_warmup_token_denominator_and_lifecycle_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture"
            rec, _ = create_fixture(output)
            row = summarize_run(output)
            self.assertEqual(row["kind"], "cpu_fixture")
            self.assertEqual(row["measured_steps"], 2)
            self.assertEqual(row["generated_tokens"], 24)
            self.assertEqual(row["trajectories"], 4)
            self.assertEqual(row["syncs_per_rollout"], 3)
            total_s = (
                sum(e["duration_ms"] for e in rec.events if e["name"] == "optimizer_step" and e["step"] > 1) / 1000
            )
            self.assertAlmostEqual(row["generated_tokens_per_s"], 24 / total_s)
            details = json.loads((output / "summary.json").read_text())
            self.assertTrue(all(r["sleep_inside"] == 3 and r["sleep_after"] == 0 for r in details["rollouts"]))

    def test_cleanup_outside_generate_is_in_rollout_lifecycle(self):
        rec = Recorder("unused")
        rec.step = 1
        rec.rollout = 0
        with rec.span("optimizer_step"):
            with rec.span("rollout_and_score"):
                with rec.span("rollout", trajectories=1):
                    pass
                with rec.span("engine_sleep"):
                    pass
        _, details = reduce_events(rec.events, 0)
        row = details["rollouts"][0]
        self.assertEqual(row["sleep_after"], 1)
        self.assertAlmostEqual(row["rollout_lifecycle_ms"], row["rollout_raw_ms"] + row["cleanup_ms"])

    def test_wrong_variant_and_invalid_config_fail(self):
        validate_lifecycle(SimpleNamespace(), "baseline")
        validate_lifecycle(SimpleNamespace(), "baseline_cache")
        with self.assertRaises(RuntimeError):
            validate_lifecycle(SimpleNamespace(), "awake")
        config = json.loads((ROOT / "HPML_WORK/profiling/configs/smoke.json").read_text())
        validate_config(config)
        broken = copy.deepcopy(config)
        broken["warmup_steps"] = broken["max_steps"]
        with self.assertRaises(ValueError):
            validate_config(broken)

    def test_failed_runs_are_not_summarized(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "manifest.json").write_text('{"status": "failed"}')
            with self.assertRaises(ValueError):
                summarize_run(path)


if __name__ == "__main__":
    unittest.main()
