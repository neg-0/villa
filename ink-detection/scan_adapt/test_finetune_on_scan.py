"""Run: python -m pytest test_finetune_on_scan.py (from ink-detection/scan_adapt). CPU only, a few seconds."""
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from finetune_on_scan import finetune, load_pair, sample  # noqa: E402


class TinyInk(torch.nn.Module):
    """Stand-in with the real model's input/output shapes: (1, 1, 62, 256, 256) -> (1, 1, 64, 64) logits."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, (62, 4, 4), stride=(62, 4, 4))
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        return self.bn(self.conv(x)[:, :, 0])


def _write_pair(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.zeros((320, 320), bool)
    for _ in range(12):
        y, x = rng.integers(0, 300, 2)
        lab[y:y + 12, x:x + 40] = True
    ct = rng.integers(60, 100, (70, 320, 320)).astype(np.uint8)
    ct[:, lab] += 60  # ink is brighter through the stack, so it is learnable
    np.save(tmp_path / "stack.npy", ct)
    Image.fromarray((lab * 255).astype(np.uint8)).save(tmp_path / "labels.png")
    return tmp_path / "stack.npy", tmp_path / "labels.png"


def test_load_pair_and_sample_shapes(tmp_path):
    s, l = _write_pair(tmp_path)
    ct, lab, m = load_pair(s, l, near_px=50)
    assert ct.shape == (62, 320, 320) and lab.shape == m.shape == (320, 320)
    X, Y, M = sample([(ct, lab, m)], np.random.default_rng(0))
    assert X.shape == (62, 256, 256) and Y.shape == M.shape == (64, 64)
    assert 0 <= X.min() and X.max() <= 1 and Y.mean() > 0


def test_finetune_learns_saves_and_resumes(tmp_path):
    s, l = _write_pair(tmp_path)
    pairs = [load_pair(s, l, near_px=50)]
    torch.manual_seed(0)
    net = TinyInk()
    info = finetune(net, pairs, tmp_path / "out", torch.device("cpu"), iters=60, lr=5e-2, save_every=30, log_every=30)
    assert info["skipped_nonfinite"] == 0
    assert np.mean(info["loss"][-10:]) < np.mean(info["loss"][:10])
    state = torch.load(tmp_path / "out" / "final.pt", weights_only=False)["state_dict"]
    assert set(state) == set(net.state_dict())
    assert json.loads((tmp_path / "out" / "train_log.json").read_text())["iters"] == 60

    # A rerun resumes from the iteration checkpoint instead of starting over.
    (tmp_path / "out" / "final.pt").unlink()
    info2 = finetune(TinyInk(), pairs, tmp_path / "out", torch.device("cpu"), iters=60, lr=5e-2, save_every=30)
    assert info2["loss"] == info["loss"]
