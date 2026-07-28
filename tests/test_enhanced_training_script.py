import os
from pathlib import Path
import shlex
import stat
import subprocess
import tempfile
import unittest


class EnhancedTrainingScriptTest(unittest.TestCase):
    def test_launcher_invokes_training_with_all_features(self):
        repository_root = Path(__file__).resolve().parents[1]
        launcher = (
            repository_root
            / "2024-CVPR-RDE"
            / "run_rde_amp_ema_gc.sh"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            invocation_file = temp_path / "invocation.txt"
            fake_uv = temp_path / "uv"
            fake_uv.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" > "$RDE_TEST_INVOCATION"\n',
                encoding="utf-8",
            )
            fake_uv.chmod(
                fake_uv.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
            environment = os.environ.copy()
            environment["PATH"] = (
                f"{temp_path}{os.pathsep}{environment['PATH']}"
            )
            environment["RDE_TEST_INVOCATION"] = str(invocation_file)

            subprocess.run(
                ["bash", str(launcher)],
                cwd=repository_root,
                env=environment,
                check=True,
            )
            invocation = shlex.split(
                invocation_file.read_text(encoding="utf-8")
            )

        self.assertEqual(invocation[:3], ["run", "python", "train.py"])
        self.assertIn("--amp", invocation)
        self.assertEqual(
            invocation[invocation.index("--amp_dtype") + 1],
            "fp16",
        )
        self.assertIn("--ema", invocation)
        self.assertEqual(
            invocation[invocation.index("--ema_decay") + 1],
            "0.999",
        )
        self.assertIn("--gradient_checkpointing", invocation)


if __name__ == "__main__":
    unittest.main()
