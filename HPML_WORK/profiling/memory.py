"""Optional NVML device-wide sampling, kept separate from PyTorch allocator statistics."""

import csv
import threading


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MemorySampler:
    def __init__(self, torch, recorder, interval_ms=100, uuid=None):
        self.torch, self.recorder = torch, recorder
        self.interval_ms, self.uuid = interval_ms, uuid
        self.rows = []
        self.status = {"status": "disabled", "interval_ms": interval_ms}
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None

    def start(self):
        if self.interval_ms == 0:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            uuid = self.uuid or getattr(self.torch.cuda.get_device_properties(0), "uuid", None)
            if uuid is None:
                raise RuntimeError("CUDA UUID unavailable; pass --nvml-uuid (do not guess physical GPU index)")
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            self.status.update(status="sampling", uuid=str(uuid))

            def sample():
                while not self._stop.is_set():
                    try:
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        self.rows.append(
                            dict(
                                time_ns=self.recorder.now_ns(),
                                phase=self.recorder.phase,
                                step=self.recorder.step,
                                used_bytes=memory.used,
                                total_bytes=memory.total,
                            )
                        )
                    except Exception as exc:
                        self.status.update(status="partial", reason=str(exc))
                        break
                    self._stop.wait(self.interval_ms / 1000)

            self._thread = threading.Thread(target=sample, daemon=True, name="ann-nvml")
            self._thread.start()
        except Exception as exc:
            self.status.update(status="unavailable", reason=str(exc))

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self.status["status"] == "sampling":
            self.status["status"] = "complete"
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                self.status["shutdown_error"] = str(exc)
