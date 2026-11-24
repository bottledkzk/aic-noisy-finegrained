import csv
import io
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from aic_finegrained.cli import main

VISION_INSTALLED = all(find_spec(module) for module in ("torch", "torchvision", "transformers", "PIL"))


@unittest.skipUnless(VISION_INSTALLED, "Install the vision extra to run image tests")
class ImageTests(unittest.TestCase):
    def test_group_excluded_anchor_never_uses_duplicate_pair(self):
        import torch

        from aic_finegrained.groupcache import build_groups, posterior, soften

        vec = torch.nn.functional.normalize(torch.tensor([
            [1.0, 0.0, 0.0], [1.0, 0.001, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0],
        ]))
        groups, _ = build_groups(vec, device="cpu")
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(np.unique(groups)), 4)
        labels = torch.tensor([0, 0, 1, 2, 2])
        probs, removed = posterior(vec, labels, torch.from_numpy(groups), 3, k=2)
        self.assertEqual(removed, 2)
        self.assertEqual(float(probs[0, 0]), 0.0)
        self.assertEqual(float(soften(probs, labels)[0][0, 0]), 0.5)

    def test_tiny_clip_training_reload_inference_and_replay(self):
        with TemporaryDirectory(prefix="aic-image-test-") as temp:
            root = Path(temp) / "demo"
            with redirect_stdout(io.StringIO()):
                main(["demo-images", "--out", str(root)])
                main(["scores", "--input", str(root / "canonical.npy"), "--out", str(root / "q.npy")])
            self.assertEqual(len(list((root / "training").glob("*.pt"))), 5)
            self.assertEqual(np.load(root / "canonical.npy").shape, (6, 3))
            with (root / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["path"] for row in rows], [f"item-{i:02d}.png" for i in range(12, 18)])
            probs = np.load(root / "q.npy")
            np.testing.assert_allclose(probs.sum(1), 1, atol=1e-12)
            self.assertEqual([row["label"] for row in rows], [f"class-{i}" for i in probs.argmax(1)])
