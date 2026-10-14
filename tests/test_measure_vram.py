import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import processor.processor as processor


class StubPoller:
    created = []
    stopped = []

    def __init__(self, log_dir, name="system_memory", **kwargs):
        StubPoller.created.append((str(log_dir), name))
        self.name = name
        self.gib = 5.0 if name == "gmm_system_memory" else 6.0

    def start(self):
        return self

    def peaks(self):
        return {"system_process_peak_gib": self.gib, "system_gpu_peak_gib": 8.0, "system_poll_samples": 3}

    def stop(self):
        StubPoller.stopped.append(self.name)
        return self.peaks()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc_info):
        self.stop()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch):
        return {"bge_loss": self.linear(batch["images"]).pow(2).mean()}


class RecordingSession:
    def __init__(self):
        self.logged = []
        self.summary = {}

    def log(self, metrics, step=None):
        self.logged.append(metrics)

    def set_summary(self, metrics):
        self.summary.update(metrics)

    def save(self, path, base_path=None):
        pass


class MeasureVramTest(unittest.TestCase):
    def test_runs_exact_steps_and_writes_report(self):
        model = TinyModel()
        loader = [
            {"images": torch.ones(2, 2), "index": torch.tensor([0, 1])}
            for _ in range(3)
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        calls = []
        cpu = torch.device("cpu")
        StubPoller.created.clear()
        with tempfile.TemporaryDirectory() as out, \
                patch.object(processor, "get_loss", lambda *a, **k: (torch.ones(2), torch.ones(2))), \
                patch.object(processor.torch, "device", lambda name: cpu), \
                patch.object(processor, "SystemMemoryPoller", StubPoller), \
                patch.object(torch.cuda, "synchronize", lambda device=None: calls.append("sync")), \
                patch.object(torch.cuda, "empty_cache", lambda: calls.append("empty")), \
                patch.object(torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append("reset")), \
                patch.object(torch.cuda, "memory_allocated", lambda device=None: 512 * 1024 ** 2), \
                patch.object(torch.cuda, "max_memory_allocated", lambda device=None: 1024 ** 3), \
                patch.object(torch.cuda, "max_memory_reserved", lambda device=None: 2 * 1024 ** 3):
            args = SimpleNamespace(
                measure_vram_steps=150, amp=False, amp_dtype="fp16",
                log_period=100, name="RDE", batch_size=2,
                gradient_checkpointing=False, output_dir=out,
            )
            session = RecordingSession()
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            report = processor.do_measure_vram(
                args, model, loader, optimizer,
                wandb_session=session, scaler=scaler,
            )
            with open(f"{out}/vram_measure.json") as f:
                saved = json.load(f)
        self.assertEqual(calls[:5], ["sync", "reset", "sync", "sync", "empty"])
        self.assertEqual(calls[5], "reset")
        self.assertEqual(StubPoller.created, [(out, "gmm_system_memory"), (out, "system_memory")])
        self.assertEqual(saved["system_process_peak_gib"], 6.0)
        self.assertEqual(report["steps"], 150)
        self.assertEqual(sorted(saved["checkpoints"]), ["100", "150"])
        self.assertEqual(saved["gmm_labeling"]["max_memory_allocated_gib"], 1.0)
        self.assertEqual(saved["gmm_labeling"]["baseline_memory_allocated_gib"], 0.5)
        self.assertEqual(saved["baseline_memory_allocated_gib"], 0.5)
        self.assertEqual(session.summary["vram_measure/gmm_labeling/baseline_memory_allocated_gib"], 0.5)
        self.assertEqual(saved["gmm_labeling"]["system_process_peak_gib"], 5.0)
        self.assertEqual(saved["system_gpu_peak_gib"], 8.0)
        self.assertEqual(saved["max_memory_allocated_gib"], 1.0)
        self.assertEqual(saved["max_memory_reserved_gib"], 2.0)
        self.assertEqual(session.summary["vram_measure/steps"], 150)


    def run_failing(self, get_loss, model):
        cpu = torch.device("cpu")
        StubPoller.created.clear()
        StubPoller.stopped.clear()
        loader = [{"images": torch.ones(2, 2), "index": torch.tensor([0, 1])}]
        with tempfile.TemporaryDirectory() as out, \
                patch.object(processor, "get_loss", get_loss), \
                patch.object(processor.torch, "device", lambda name: cpu), \
                patch.object(processor, "SystemMemoryPoller", StubPoller), \
                patch.object(torch.cuda, "synchronize", lambda device=None: None), \
                patch.object(torch.cuda, "empty_cache", lambda: None), \
                patch.object(torch.cuda, "reset_peak_memory_stats", lambda device=None: None), \
                patch.object(torch.cuda, "memory_allocated", lambda device=None: 0), \
                patch.object(torch.cuda, "max_memory_allocated", lambda device=None: 0), \
                patch.object(torch.cuda, "max_memory_reserved", lambda device=None: 0):
            args = SimpleNamespace(
                measure_vram_steps=5, amp=False, amp_dtype="fp16",
                log_period=100, name="RDE", batch_size=2,
                gradient_checkpointing=False, output_dir=out,
            )
            with self.assertRaises(RuntimeError):
                processor.do_measure_vram(
                    args, model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
                    wandb_session=RecordingSession(),
                    scaler=torch.amp.GradScaler("cpu", enabled=False),
                )

    def test_exception_in_gmm_pass_stops_poller(self):
        def failing_get_loss(*args, **kwargs):
            raise RuntimeError("oom")

        self.run_failing(failing_get_loss, TinyModel())
        self.assertEqual(StubPoller.stopped, ["gmm_system_memory"])

    def test_exception_in_training_stops_poller(self):
        class FailingModel(TinyModel):
            def forward(self, batch):
                raise RuntimeError("oom")

        self.run_failing(lambda *a, **k: (torch.ones(2), torch.ones(2)), FailingModel())
        self.assertEqual(StubPoller.stopped, ["gmm_system_memory", "gmm_system_memory", "system_memory"])


if __name__ == "__main__":
    unittest.main()
