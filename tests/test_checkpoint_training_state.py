import os
import tempfile
import unittest

import torch

from utils.checkpoint import Checkpointer
from utils.training import build_ema_model


class StatefulScaler:
    def __init__(self, scale):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = state["scale"]


class CheckpointTrainingStateTest(unittest.TestCase):
    def test_save_and_resume_round_trips_raw_ema_and_scaler_state(self):
        with tempfile.TemporaryDirectory() as save_dir:
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(1.0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            scaler = StatefulScaler(scale=1024.0)
            ema_model = build_ema_model(model, decay=0.5)
            ema_model.module.weight.data.fill_(5.0)
            ema_model.n_averaged.fill_(7)
            checkpointer = Checkpointer(
                model,
                optimizer,
                scheduler,
                save_dir,
                True,
                scaler=scaler,
                ema_model=ema_model,
            )
            checkpointer.save("state", epoch=3)

            model.weight.data.fill_(9.0)
            ema_model.module.weight.data.fill_(9.0)
            ema_model.n_averaged.zero_()
            scaler.scale = 1.0
            checkpoint = checkpointer.resume(
                os.path.join(save_dir, "state.pth")
            )

        self.assertEqual(model.weight.item(), 1.0)
        self.assertEqual(ema_model.module.weight.item(), 5.0)
        self.assertEqual(ema_model.n_averaged.item(), 7)
        self.assertEqual(scaler.scale, 1024.0)
        self.assertEqual(checkpoint["epoch"], 3)

    def test_load_prefers_ema_model_weights_when_present(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint_file:
            raw_model = torch.nn.Linear(1, 1, bias=False)
            raw_model.weight.data.fill_(1.0)
            ema_model = build_ema_model(raw_model, decay=0.5)
            ema_model.module.weight.data.fill_(5.0)
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                },
                checkpoint_file.name,
            )
            target = torch.nn.Linear(1, 1, bias=False)

            Checkpointer(target).load(
                checkpoint_file.name,
                prefer_ema=True,
            )

        self.assertEqual(target.weight.item(), 5.0)

    def test_preferred_ema_load_falls_back_to_legacy_model_state(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint_file:
            raw_model = torch.nn.Linear(1, 1, bias=False)
            raw_model.weight.data.fill_(2.0)
            torch.save(
                {"model": raw_model.state_dict()},
                checkpoint_file.name,
            )
            target = torch.nn.Linear(1, 1, bias=False)

            Checkpointer(target).load(
                checkpoint_file.name,
                prefer_ema=True,
            )

        self.assertEqual(target.weight.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
