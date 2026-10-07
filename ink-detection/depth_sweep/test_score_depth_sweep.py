"""Run: python -m pytest test_score_depth_sweep.py (from ink-detection/depth_sweep)."""
import numpy as np
from PIL import Image

from scipy import ndimage

from score_depth_sweep import align_labels, load_map, main, score


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
    assert res["label_origin"] == [20, 30]
    assert res["label_best"] == -8
    assert res["offsets"][-8]["auc"] > res["offsets"][-8]["auc_at_origin"]


def test_label_search_uses_one_translation_for_every_offset():
    """A per-map search lets a map with no ink fit noise; one shared translation does not (V-025)."""
    rng = np.random.default_rng(3)
    labels = (ndimage.gaussian_filter(rng.random((200, 200)), 3) > 0.53).astype(np.float32)
    ink = np.clip(0.3 + 0.5 * labels[50:130, 50:130] + 0.1 * rng.standard_normal((80, 80)), 0, 1)
    noise = np.clip(ndimage.gaussian_filter(rng.random((80, 80)), 3) * 2 - 0.5, 0, 1)
    res = score({0: ink, 32: noise}, labels, label_origin=(46, 46), label_search=16)
    assert res["label_origin"] == [50, 50] and res["label_best"] == 0
    assert abs(res["offsets"][32]["auc"] - 0.5) < 0.1
    per_map = max(score({32: noise}, labels, label_origin=(46, 46), label_search=16)["offsets"][32]["auc"], 0.5)
    assert per_map > res["offsets"][32]["auc"]


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


def test_large_label_search_runs_coarse_to_fine():
    rng = np.random.default_rng(1)
    labels = (ndimage.gaussian_filter(rng.random((400, 500)), 6) > 0.52).astype(np.float32)
    pred = np.clip(labels[150:310, 180:420] * 0.6 + 0.2 + 0.05 * rng.standard_normal((160, 240)), 0, 1)
    L, at = align_labels(pred, labels, origin=(100, 100), search=120)
    assert at == (150, 180) and L.shape == pred.shape


def test_zarr_labels_window(tmp_path):
    import zarr

    lab = np.zeros((50, 60), np.uint8)
    lab[10:20, 30:40] = 255
    g = zarr.open_group(str(tmp_path / "inklabels.zarr"), mode="w")
    g.create_dataset("0", data=lab, chunks=(16, 16))
    a = load_map(tmp_path / "inklabels.zarr", window=(5, 25, 25, 45))
    assert a.shape == (20, 20) and a.max() == 1.0 and a[5:15, 5:15].all() and a.sum() == 100
