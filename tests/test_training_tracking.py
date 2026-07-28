import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from processor.processor import do_train


class RecordingSession:
    enabled = True

    def __init__(self):
        self.payloads = []

    def log(self, payload, step=None):
        self.payloads.append(payload)


class BatchValue:
    def __init__(self, shape=()):
        self.shape = shape

    def to(self, device):
        return self


class IndexValue:
    def to(self, device):
        return self

    def cpu(self):
        return torch.tensor([0, 1])


class OneEpochModel:
    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        return {
            "bge_loss": torch.tensor(1.0, requires_grad=True),
            "tse_loss": torch.tensor(2.0, requires_grad=True),
            "temperature": torch.tensor(0.02),
        }


class TrainingTrackingTest(unittest.TestCase):
    def test_one_epoch_logs_measured_train_and_validation_metrics(self):
        train_loader = [
            {
                "images": BatchValue(shape=(2, 3, 4, 4)),
                "index": IndexValue(),
            }
        ]
        args = SimpleNamespace(
            log_period=100,
            eval_period=1,
            num_epoch=1,
            output_dir=tempfile.gettempdir(),
            distributed=False,
        )
        optimizer = Mock()
        scheduler = Mock()
        scheduler.get_lr.return_value = [1e-5]
        checkpointer = Mock()
        session = RecordingSession()

        with (
            patch("processor.processor.SummaryWriter"),
            patch("processor.processor.synchronize"),
            patch("processor.processor.get_rank", return_value=0),
            patch("processor.processor.get_world_size", return_value=1, create=True),
            patch(
                "processor.processor.get_loss",
                return_value=(torch.ones(2), torch.ones(2)),
            ),
            patch(
                "processor.processor.start_measurement",
                return_value=10.0,
                create=True,
            ),
            patch(
                "processor.processor.finish_cuda_timer",
                return_value=5.0,
                create=True,
            ),
            patch(
                "processor.processor.get_peak_vram_metrics",
                return_value={"peak_vram_allocated_mb": 9000.0},
                create=True,
            ),
            patch(
                "processor.processor.get_global_processed_examples",
                return_value=2,
                create=True,
            ),
            patch(
                "processor.processor._evaluate_with_efficiency",
                return_value=(
                    {
                        "t2i_R1": 75.94,
                        "i2t_R1": 86.0,
                        "bge_t2i_R1": 73.49,
                    },
                    {"epoch_seconds": 2.0},
                    {"peak_vram_allocated_mb": 7000.0},
                ),
                create=True,
            ),
            patch("processor.processor.torch.cuda.empty_cache"),
        ):
            best_top1, best_epoch = do_train(
                start_epoch=1,
                args=args,
                model=OneEpochModel(),
                train_loader=train_loader,
                evaluator=Mock(),
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                wandb_session=session,
            )

        train_payload, val_payload = session.payloads
        self.assertEqual(train_payload["train/loss"], 3.0)
        self.assertEqual(train_payload["train/examples_per_second"], 0.4)
        self.assertEqual(val_payload["val/t2i_R1"], 75.94)
        self.assertEqual(val_payload["val/bge_t2i_R1"], 73.49)
        self.assertEqual((best_top1, best_epoch), (75.94, 1))
        checkpointer.save.assert_any_call(
            "best",
            num_epoch=1,
            iteration=0,
            epoch=1,
        )


if __name__ == "__main__":
    unittest.main()
