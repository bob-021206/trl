"""Nested wall-clock spans; no CUDA dependency, GPU time is never inferred from wall time."""

import json
import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


class Recorder:
    def __init__(self, output, synchronize=None, ranges=None):
        self.output = Path(output)
        self.synchronize = synchronize
        self.ranges = ranges or (lambda name: nullcontext())
        self.events = []
        self.stack = []
        self.phase = "train"
        self.step = None
        self.rollout = None
        self.turn = None
        self._next_id = 0
        self._origin = time.perf_counter_ns()
        self.origin_unix_ns = time.time_ns()
        self._ranges = {}

    def now_ns(self):
        return time.perf_counter_ns() - self._origin

    def begin(self, name, **fields):
        if self.synchronize:
            self.synchronize()
        event = {
            "event_id": self._next_id,
            "parent_id": self.stack[-1]["event_id"] if self.stack else None,
            "name": name,
            "phase": self.phase,
            "step": self.step,
            "rollout": self.rollout,
            "turn": self.turn,
            "start_ns": time.perf_counter_ns() - self._origin,
            "status": "ok",
            "fields": fields,
            "_children_ns": 0,
        }
        self._next_id += 1
        region = self.ranges(name)
        region.__enter__()
        self._ranges[event["event_id"]] = region
        self.stack.append(event)
        return event

    def end(self, event, error=None):
        if not self.stack or self.stack[-1] is not event:
            raise RuntimeError("Spans must close in LIFO order")
        if self.synchronize and error is None:
            self.synchronize()
        self._ranges.pop(event["event_id"]).__exit__(
            type(error) if error else None, error, error.__traceback__ if error else None
        )
        elapsed = time.perf_counter_ns() - self._origin - event["start_ns"]
        self.stack.pop()
        event["duration_ms"] = elapsed / 1e6
        event["exclusive_ms"] = max(0, elapsed - event.pop("_children_ns")) / 1e6
        if error is not None:
            event["status"] = "error"
            event["fields"]["error_type"] = type(error).__name__
        if self.stack:
            self.stack[-1]["_children_ns"] += elapsed
        self.events.append(event)

    @contextmanager
    def span(self, name, **fields):
        event = self.begin(name, **fields)
        error = None
        try:
            yield event["fields"]
        except BaseException as exc:
            error = exc
            raise
        finally:
            # Trainer callbacks may leave a step/update range open if training raises.
            # Preserve the original failure while closing those children first.
            if error is not None:
                while self.stack and self.stack[-1] is not event:
                    self.end(self.stack[-1], error)
            self.end(event, error)

    def close_open(self, error):
        while self.stack:
            self.end(self.stack[-1], error)

    def flush(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as stream:
            for event in sorted(self.events, key=lambda item: item["event_id"]):
                stream.write(json.dumps(json_value(event), allow_nan=False) + "\n")


def json_value(value):
    """Undefined trainer metrics become JSON null, never nonstandard NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
