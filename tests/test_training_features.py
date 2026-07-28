import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model.CrossEmbeddingLayer_tse import (
    TexualEmbeddingLayer,
    VisualEmbeddingLayer,
)
from model.clip_model import Transformer
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

    def test_transformer_checkpoints_only_during_gradient_enabled_training(self):
        transformer = Transformer(width=8, layers=2, heads=1)
        transformer.set_gradient_checkpointing(True)
        transformer.train()

        def run_block(function, *args, **kwargs):
            self.assertFalse(kwargs["use_reentrant"])
            return function(*args)

        with patch(
            "model.clip_model.checkpoint",
            side_effect=run_block,
        ) as checkpoint_call:
            transformer([torch.randn(3, 2, 8, requires_grad=True)])
            self.assertEqual(checkpoint_call.call_count, 2)

            transformer.eval()
            transformer([torch.randn(3, 2, 8, requires_grad=True)])
            self.assertEqual(checkpoint_call.call_count, 2)

            transformer.train()
            with torch.no_grad():
                transformer([torch.randn(3, 2, 8)])
            self.assertEqual(checkpoint_call.call_count, 2)

    def test_checkpointed_transformer_matches_regular_forward_and_backward(self):
        regular = Transformer(width=8, layers=2, heads=1)
        checkpointed = Transformer(width=8, layers=2, heads=1)
        checkpointed.load_state_dict(regular.state_dict())
        checkpointed.set_gradient_checkpointing(True)
        regular.train()
        checkpointed.train()
        regular_input = torch.randn(3, 2, 8, requires_grad=True)
        checkpointed_input = regular_input.detach().clone().requires_grad_(True)

        regular_output = regular([regular_input])[0]
        checkpointed_output = checkpointed([checkpointed_input])[0]
        regular_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, regular_output)
        torch.testing.assert_close(
            checkpointed_input.grad,
            regular_input.grad,
        )

    def test_tse_layers_accept_inputs_matching_fp32_model_parameters(self):
        text_layer = TexualEmbeddingLayer(
            input_dim=4,
            embed_dim=4,
            ratio=0.5,
        )
        visual_layer = VisualEmbeddingLayer(
            input_dim=4,
            embed_dim=4,
            ratio=0.5,
        )
        features = torch.randn(2, 4, 4)
        attention = torch.randn(2, 4, 4)
        text = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 0]])

        text_output = text_layer(features, text, attention.clone())
        visual_output = visual_layer(features, attention.clone())

        self.assertEqual(text_output.dtype, torch.float32)
        self.assertEqual(visual_output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
