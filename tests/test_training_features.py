import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils.options import get_args
from utils.training import (
    build_ema_model,
    get_autocast_dtype,
    optimizer_step,
    validate_training_options,
)


class FakeScaler:
    def __init__(self, old_scale, new_scale):
        self.old_scale = old_scale
        self.new_scale = new_scale
        self._updated = False

    def scale(self, loss):
        return loss

    def get_scale(self):
        return self.new_scale if self._updated else self.old_scale

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self._updated = True


class TrainingFeatureTest(unittest.TestCase):
    def test_training_feature_options_are_disabled_by_default(self):
        with patch.object(sys, "argv", ["train.py"]):
            args = get_args()

        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "fp16")
        self.assertFalse(args.ema)
        self.assertEqual(args.ema_decay, 0.999)
        self.assertFalse(args.gradient_checkpointing)

    def test_training_feature_options_can_be_enabled(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--amp",
                "--amp_dtype",
                "bf16",
                "--ema",
                "--ema_decay",
                "0.995",
                "--gradient_checkpointing",
            ],
        ):
            args = get_args()

        self.assertTrue(args.amp)
        self.assertEqual(args.amp_dtype, "bf16")
        self.assertTrue(args.ema)
        self.assertEqual(args.ema_decay, 0.995)
        self.assertTrue(args.gradient_checkpointing)

    def test_ema_decay_must_be_strictly_between_zero_and_one(self):
        for value in ("0", "1", "-0.1", "1.1"):
            with (
                self.subTest(value=value),
                patch.object(
                    sys,
                    "argv",
                    ["train.py", "--ema_decay", value],
                ),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                get_args()

    def test_bf16_amp_requires_supported_cuda_device(self):
        args = SimpleNamespace(amp=True, amp_dtype="bf16")

        with patch(
            "utils.training.torch.cuda.is_bf16_supported",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "BF16 AMP"):
                validate_training_options(args)

    def test_amp_dtype_names_map_to_torch_dtypes(self):
        self.assertIs(get_autocast_dtype("fp16"), torch.float16)
        self.assertIs(get_autocast_dtype("bf16"), torch.bfloat16)

    def test_optimizer_step_reports_success_when_scale_does_not_drop(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = FakeScaler(old_scale=8.0, new_scale=8.0)

        stepped = optimizer_step(parameter.square(), optimizer, scaler)

        self.assertTrue(stepped)
        self.assertAlmostEqual(parameter.item(), 0.8, places=6)

    def test_optimizer_step_reports_overflow_when_scale_drops(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = FakeScaler(old_scale=8.0, new_scale=4.0)

        stepped = optimizer_step(parameter.square(), optimizer, scaler)

        self.assertFalse(stepped)

    def test_ema_model_uses_requested_decay_for_parameters_and_buffers(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=False),
            torch.nn.BatchNorm1d(1),
        )
        model[0].weight.data.fill_(1.0)
        model[1].running_mean.fill_(1.0)
        ema_model = build_ema_model(model, decay=0.5)

        model[0].weight.data.fill_(3.0)
        model[1].running_mean.fill_(3.0)
        ema_model.update_parameters(model)
        model[0].weight.data.fill_(5.0)
        model[1].running_mean.fill_(5.0)
        ema_model.update_parameters(model)

        self.assertEqual(ema_model.module[0].weight.item(), 4.0)
        self.assertEqual(ema_model.module[1].running_mean.item(), 4.0)


if __name__ == "__main__":
    unittest.main()
