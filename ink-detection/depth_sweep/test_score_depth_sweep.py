"""Run: python -m pytest test_score_depth_sweep.py (from ink-detection/depth_sweep)."""
import numpy as np
from PIL import Image

from score_depth_sweep import main, score


def _maps(true_offset=-8, shape=(64, 96), seed=0):
    """Labels with ink strokes, and maps that show them most sharply at true_offset."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.float32)
    labels[10:20, 10:80] = 1
    labels[40:50, 20:60] = 1
    maps = {}
    for o in (-16, -8, 0, 8, 16):
        signal = max(0.0, 1 - abs(o - true_offset) / 16)
        maps[o] = np.clip(0.3 + 0.5 * signal * labels + 0.1 * rng.standard_normal(shape), 0, 1).astype(np.float32)
    return maps, labels


def test_picks_the_offset_with_the_ink():
    maps, labels = _maps()
    res = score(maps, labels)
    assert res["label_free_pick"] == -8
    assert res["label_best"] == -8
    assert res["offsets"][-8]["auc"] > 0.99 > res["offsets"][16]["auc"]


def test_label_search_recovers_the_map_position():
    maps, labels = _maps()
    big = np.zeros((100, 140), np.float32)
    big[20:84, 30:126] = labels
    res = score(maps, big, label_origin=(16, 26), label_search=8)
    assert res["offsets"][-8]["label_origin"] == [20, 30]
    assert res["label_best"] == -8


def test_cli_reads_png(tmp_path, capsys):
    maps, labels = _maps()
    args = []
    for o, m in maps.items():
        p = tmp_path / f"o{o}.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(p)
        args += ["--map", str(o), str(p)]
    Image.fromarray((labels * 255).astype(np.uint8)).save(tmp_path / "labels.png")
    assert main(args + ["--labels", str(tmp_path / "labels.png")]) == 0
    assert "label-free pick: -8 | label best: -8" in capsys.readouterr().out
