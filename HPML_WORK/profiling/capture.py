"""Lecture 3: scheduled torch.profiler, operator tables, CUDA events and memory snapshots."""

import csv
from contextlib import contextmanager, nullcontext
from pathlib import Path


def make_ranges(torch, mode):
    @contextmanager
    def region(name):
        with torch.profiler.record_function(name) if mode == "profile" else nullcontext():
            if mode == "nsight":
                torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                if mode == "nsight":
                    torch.cuda.nvtx.range_pop()

    return region


def export_trace(profiler, directory):
    """Public profiler APIs; self_device_time_total is the newer spelling of Self CUDA."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"train-step-{profiler.step_num}"
    profiler.export_chrome_trace(str(stem) + ".json")
    averages = profiler.key_averages()
    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=30)
    device_table = averages.table(sort_by="self_device_time_total", row_limit=30)
    Path(str(stem) + "-cpu.txt").write_text(cpu_table)
    Path(str(stem) + "-cuda.txt").write_text(device_table)
    columns = [
        "name",
        "count",
        "self_cpu_us",
        "cpu_total_us",
        "self_device_us",
        "device_total_us",
        "self_cpu_memory_bytes",
        "self_device_memory_bytes",
    ]
    with Path(str(stem) + "-operators.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for event in averages:
            writer.writerow(
                dict(
                    zip(
                        columns,
                        [
                            event.key,
                            event.count,
                            event.self_cpu_time_total,
                            event.cpu_time_total,
                            getattr(event, "self_device_time_total", 0),
                            getattr(event, "device_time_total", 0),
                            event.self_cpu_memory_usage,
                            getattr(event, "self_device_memory_usage", 0),
                        ],
                        strict=True,
                    )
                )
            )
    print(cpu_table)
    print(device_table)
    print(f"Perfetto trace: {stem}.json")


def profiler_context(torch, config, output, *, cuda=True):
    if config["mode"] != "profile":
        return nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            skip_first=config["warmup_steps"],
            wait=config["profile_wait"],
            warmup=config["profile_warmup"],
            active=config["profile_active"],
            repeat=1,
        ),
        on_trace_ready=lambda prof: export_trace(prof, output / "traces"),
        record_shapes=True,
        profile_memory=True,
        with_stack=config["with_stack"],
    )


@contextmanager
def memory_history(torch, config, output):
    if not config["memory_snapshot"]:
        yield
        return
    # These are the same version-sensitive APIs used in lecture 3.
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(str(output / "memory-snapshot.pickle"))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


def measurement_callback(torch, TrainerCallback, recorder, config, profiler_ref, allocator_samples):
    class MeasurementCallback(TrainerCallback):
        active = None
        optimizer = None
        start_event = None

        def sample_allocator(self, boundary):
            allocator_samples.append(
                dict(
                    time_ns=recorder.now_ns(),
                    phase=recorder.phase,
                    step=recorder.step,
                    boundary=boundary,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                )
            )

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            recorder.step = state.global_step + 1
            torch.cuda.reset_peak_memory_stats()
            self.sample_allocator("step_begin")
            self.active = recorder.begin("optimizer_step")
            if config["cuda_events"]:
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.optimizer = recorder.begin("optimizer_update")

        def on_optimizer_step(self, args, state, control, **kwargs):
            if self.optimizer is not None:
                recorder.end(self.optimizer)
                self.optimizer = None

        def on_step_end(self, args, state, control, **kwargs):
            end_event = None
            if config["cuda_events"]:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
            torch.cuda.synchronize()
            # End step wall time before collecting scalar stats or exporting a trace.
            recorder.end(self.active)
            self.active["fields"].update(
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_stream_elapsed_ms=self.start_event.elapsed_time(end_event) if end_event else None,
            )
            self.sample_allocator("step_end")
            self.active = None
            recorder.step = None
            if profiler_ref[0] is not None:
                profiler_ref[0].step()

        def on_log(self, args, state, control, logs=None, **kwargs):
            with recorder.span("trainer_log", global_step=state.global_step, logs=logs or {}):
                pass

    return MeasurementCallback()
