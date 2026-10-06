"""Run: python -m pytest test_diagnose_segment.py (from ink-detection/scan_diagnosis). CPU only, no model needed."""
import numpy as np
import pytest

from diagnose_segment import diagnose, findings, load_stack, window, write_markdown
from test_check_placement import _texture, _write_ome  # foundation/volume-registration, put on sys.path above


def _labels(shape=(96, 128)):
    lab = np.zeros(shape, np.float32)
    lab[10:24, 10:110] = 1
    lab[50:64, 30:90] = 1
    return lab


def _fake_predictor(labels, ink_offset, strength=0.5, seed=0):
    """Predicts ink most sharply when the window is centred ink_offset layers from the middle of the stack."""
    rng = np.random.default_rng(seed)

    def predict(stack, start, end):
        o = start - (stack.shape[2] // 2 - 31)
        s = strength * max(0.0, 1 - abs(o - ink_offset) / 16)
        return np.clip(0.3 + s * labels + 0.1 * rng.standard_normal(labels.shape), 0, 1).astype(np.float32)

    return predict


def test_window_and_load_stack(tmp_path):
    assert window(100, 0) == (19, 81) and window(100, -8) == (11, 73)
    with pytest.raises(ValueError):
        window(70, 8)
    a = np.arange(5 * 6 * 7, dtype=np.uint8).reshape(5, 6, 7)
    np.save(tmp_path / "s.npy", a)
    s = load_stack(tmp_path / "s.npy", crop=(1, 2, 3, 4))
    assert s.shape == (3, 4, 5) and s[0, 0, 0] == a[0, 1, 2] and s[0, 0, 4] == a[4, 1, 2]
    np.save(tmp_path / "w.npy", a.astype(np.uint16))
    with pytest.raises(ValueError):
        load_stack(tmp_path / "w.npy")


def test_depth_offset_and_model_findings(tmp_path):
    lab = _labels()
    stack = np.zeros((96, 128, 100), np.uint8)
    rep = diagnose(stack, _fake_predictor(lab, -8), [-16, -8, 0, 8], labels=lab, out_dir=tmp_path,
                   adapted_predict=_fake_predictor(lab, -8, strength=0.7, seed=1))
    assert rep["depth"]["label_best"] == -8 and rep["model"]["offset"] == -8
    assert rep["model"]["adapted_auc"] >= rep["model"]["canonical_auc"]
    assert any(f.startswith("Depth: ink reads best -8 layers") for f in rep["findings"])
    assert (tmp_path / "pred_canonical_o-8.png").exists() and (tmp_path / "pred_adapted_o-8.png").exists()
    write_markdown(rep, tmp_path / "report.md")
    assert "| -8 |" in (tmp_path / "report.md").read_text()

    # A model that sees nothing: low AUC everywhere, so the report points at adapting it.
    weak = diagnose(stack, _fake_predictor(lab, 0, strength=0.02), [-8, 0, 8], labels=lab)
    assert any("adapt it with ink-detection/scan_adapt" in f for f in weak["findings"])


def test_placement_no_match_comes_first(tmp_path):
    vol = _texture((48, 384, 384))
    ref = _write_ome(tmp_path / "ref.zarr", vol, 2.0)
    other = _write_ome(tmp_path / "other.zarr", _texture((48, 384, 384), seed=1), 2.0)
    lab = _labels()
    rep = diagnose(np.zeros((96, 128, 100), np.uint8), _fake_predictor(lab, 0), [-8, 0, 8], labels=lab,
                   reference=ref, surface_volume=other,
                   placement_opts={"size_um": 400, "search_um": 64, "windows": [(150.0, 150.0), (250.0, 200.0)]})
    assert rep["placement"]["verdict"] == "no match within the search range"
    assert rep["findings"][0].startswith("Placement: the surface does not match")


def test_findings_without_labels():
    rep = {"depth": {"offsets": {0: {"frac_above_0.5": 0.01}, 8: {"frac_above_0.5": 0.05}}, "label_free_pick": 8},
           "model": {"offset": 8}}
    f = findings(rep)
    assert len(f) == 1 and f[0].startswith("Depth: without labels, the most ink is predicted +8 layers")
