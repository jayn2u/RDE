import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.efficiency import build_epoch_efficiency_metrics
from utils.wandb_tracking import (
    WandbSession,
    log_val_metrics,
    read_env_value,
    start_train_run,
)


class RecordingSession:
    enabled = True

    def __init__(self):
        self.payloads = []

    def log(self, payload, step=None):
        self.payloads.append(payload)


class FakeRun:
    def __init__(self, project, entity):
        self.id = "rde-run-id"
        self.project = project
        self.entity = entity
        self.summary = {}
        self.defined_metrics = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, metrics, step=None):
        pass

    def save(self, path, base_path=None):
        pass

    def finish(self):
        pass


class FakeWandb:
    def __init__(self):
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return FakeRun(kwargs["project"], kwargs["entity"])


class WandbTrackingTest(unittest.TestCase):
    def test_process_environment_wins_over_env_file(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as env_file:
            env_file.write("WANDB_PROJECT=file-project\n")
            env_path = env_file.name
        self.addCleanup(os.unlink, env_path)

        with patch.dict(os.environ, {"WANDB_PROJECT": "rde"}, clear=True):
            value = read_env_value("WANDB_PROJECT", env_path)

        self.assertEqual(value, "rde")

    def test_project_falls_back_to_rde(self):
        fake_wandb = FakeWandb()
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                wandb=True,
                wandb_env_file="",
                wandb_project="",
                wandb_entity="",
                wandb_run_name="",
                wandb_group="",
                wandb_tags=[],
                wandb_notes="",
                output_dir=output_dir,
                dataset_name="CUHK-PEDES",
                loss_names="TAL",
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("utils.wandb_tracking.wandb", fake_wandb),
            ):
                session = start_train_run(args)

        self.assertTrue(session.enabled)
        self.assertEqual(fake_wandb.init_kwargs["project"], "rde")
        self.assertEqual(fake_wandb.init_kwargs["group"], "CUHK-PEDES")

    def test_primary_and_component_validation_metrics_share_payload(self):
        session = RecordingSession()

        log_val_metrics(
            session,
            epoch=3,
            metrics={"t2i_R1": 75.94, "bge_t2i_R1": 73.49},
        )

        payload = session.payloads[0]
        self.assertAlmostEqual(payload["val/t2i_error@1"], 24.06)
        self.assertEqual(payload["val/bge_t2i_R1"], 73.49)

    def test_disabled_session_is_a_no_op(self):
        session = WandbSession(None)
        log_val_metrics(session, epoch=1, metrics={"t2i_R1": 50.0})

    def test_epoch_efficiency_uses_processed_examples_and_gpu_seconds(self):
        metrics = build_epoch_efficiency_metrics(
            epoch_seconds=20.0,
            processed_examples=640,
            cumulative_seconds=40.0,
        )

        self.assertEqual(
            metrics,
            {
                "epoch_seconds": 20.0,
                "examples_per_second": 32.0,
                "cumulative_gpu_hours": 40.0 / 3600.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
