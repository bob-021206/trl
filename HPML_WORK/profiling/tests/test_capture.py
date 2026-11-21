import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from HPML_WORK.profiling.capture import make_ranges, measurement_callback, profiler_context
from HPML_WORK.profiling.memory import MemorySampler
from HPML_WORK.profiling.plot import validate_comparison
from HPML_WORK.profiling.recorder import Recorder
from HPML_WORK.profiling.smoke import create_fixture
from HPML_WORK.profiling.summarize import reduce_events


class CaptureTests(unittest.TestCase):
    def test_exception_closes_callback_ranges_without_hiding_failure(self):
        rec = Recorder("unused")
        with self.assertRaisesRegex(ValueError, "original failure"):
            with rec.span("train_loop"):
                rec.begin("optimizer_step")
                rec.begin("optimizer_update")
                raise ValueError("original failure")
        self.assertFalse(rec.stack)
        self.assertEqual(len(rec.events), 3)
        self.assertTrue(all(e["fields"]["error_type"] == "ValueError" for e in rec.events))

    def test_callback_boundaries_and_profiler_steps(self):
        cuda = Mock()
        cuda.memory_allocated.return_value = 10
        cuda.memory_reserved.return_value = 20
        cuda.max_memory_allocated.return_value = 30
        cuda.max_memory_reserved.return_value = 40
        rec, samples, profiler = Recorder("unused"), [], Mock()
        callback = measurement_callback(
            SimpleNamespace(cuda=cuda), object, rec, {"cuda_events": False}, [profiler], samples
        )
        state = SimpleNamespace(global_step=0)
        with rec.span("train_loop"):
            callback.on_step_begin(None, state, None)
            with rec.span("train_microstep"):
                pass
            callback.on_pre_optimizer_step(None, state, None)
            callback.on_optimizer_step(None, state, None)
            state.global_step += 1
            callback.on_step_end(None, state, None)
        step = next(e for e in rec.events if e["name"] == "optimizer_step")
        self.assertEqual(step["step"], 1)
        self.assertIsNone(rec.step)
        self.assertEqual(step["fields"]["torch_peak_allocated_bytes"], 30)
        self.assertEqual(cuda.synchronize.call_count, 2)
        self.assertEqual(profiler.step.call_count, 1)
        self.assertEqual([s["boundary"] for s in samples], ["step_begin", "step_end"])

    def test_eval_does_not_pollute_training_and_json_handles_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            rec, _ = create_fixture(Path(temp) / "fixture")
            before, details = reduce_events(rec.events, 1)
            total_steps = sum(e["duration_ms"] for e in rec.events if e["name"] == "optimizer_step" and e["step"] > 1)
            self.assertAlmostEqual(sum(details["exclusive_ms_per_step"].values()) * 2, total_steps)
            rec.phase, rec.step = "eval", 3
            with rec.span("quality_sample", correct=0, tool_calls=7, fully_followed=False):
                pass
            after, _ = reduce_events(rec.events, 1)
            self.assertEqual(before["generated_tokens_per_s"], after["generated_tokens_per_s"])
            self.assertEqual(after["accuracy"], 1)
            self.assertEqual(after["heldout_accuracy"], 0)
            with rec.span("trainer_log", global_step=3, logs={"loss": float("nan")}):
                pass
            rec.flush()
            last = json.loads(rec.output.read_text().splitlines()[-1])
            self.assertIsNone(last["fields"]["logs"]["loss"])

    def test_nvml_uses_cuda_uuid_and_degrades_gracefully(self):
        nvml = Mock()
        nvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(used=42, total=100)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(uuid=b"GPU-test"))
        )
        with patch.dict("sys.modules", {"pynvml": nvml}):
            sampler = MemorySampler(torch, Recorder("unused"), 100)
            sampler.start()
            sampler.stop()
            nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-test")
            self.assertEqual(sampler.status["status"], "complete")
            nvml.nvmlInit.side_effect = RuntimeError("driver absent")
            unavailable = MemorySampler(torch, Recorder("unused"), 100)
            unavailable.start()
            unavailable.stop()
            self.assertEqual(unavailable.status["status"], "unavailable")
            self.assertEqual(unavailable.rows, [])

    def test_comparison_rejects_incomparable_runs(self):
        base = dict(
            comparison_key="a",
            kind="gpu_training",
            mode="benchmark",
            timing_mode="throughput",
            profiled=False,
            variant="baseline",
            seed=0,
        )
        validate_comparison([base, {**base, "variant": "awake"}])
        for changes in (
            {"comparison_key": "b"},
            {"mode": "profile"},
            {"kind": "cpu_fixture"},
            {"timing_mode": "diagnostic"},
            {},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_comparison([base, {**base, **changes}])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires PyTorch (CPU is sufficient)")
    def test_real_cpu_profiler_schedule_and_artifacts(self):
        import torch

        config = dict(
            mode="profile", warmup_steps=2, profile_wait=1, profile_warmup=1, profile_active=2, with_stack=False
        )
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            directory = Path(temp)
            ranges = make_ranges(torch, "profile")
            weights = torch.randn(8, 8, requires_grad=True)
            with profiler_context(torch, config, directory, cuda=False) as prof:
                for _ in range(8):
                    with ranges("loss_forward"):
                        loss = (weights @ weights).square().mean()
                    with ranges("backward"):
                        loss.backward()
                    weights.grad = None
                    prof.step()
            traces = list((directory / "traces").glob("*.json"))
            self.assertEqual([p.name for p in traces], ["train-step-6.json"])
            trace = json.loads(traces[0].read_text())
            self.assertIn("loss_forward", {e["name"] for e in trace["traceEvents"]})
            with (directory / "traces/train-step-6-operators.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertIn("aten::mm", {r["name"] for r in rows})
            self.assertTrue((directory / "traces/train-step-6-cpu.txt").exists())


if __name__ == "__main__":
    unittest.main()
