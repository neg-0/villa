"""Run: python -m pytest test_check_placement.py (from foundation/volume-registration)."""
import numpy as np
import zarr
from scipy import ndimage

from check_placement import check_pair, shift_search


def _texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 2) * 4000 + 100


def _write_ome(path, vol, voxel_um, levels=3):
    g = zarr.open_group(str(path), mode="w")
    datasets = []
    for lv in range(levels):
        f = 2**lv
        a = vol[:, ::f, ::f] if lv else vol
        g.create_dataset(str(lv), data=a.astype(np.uint16), chunks=(16, 64, 64))
        datasets.append(
            {"path": str(lv), "coordinateTransformations": [{"type": "scale", "scale": [voxel_um, voxel_um * f, voxel_um * f]}]}
        )
    g.attrs["multiscales"] = [{"version": "0.4", "datasets": datasets}]
    return str(path)


def test_shift_search_recovers_known_shift():
    big = _texture((20, 80, 80))
    ref = big[4:16, 10:70, 10:70]
    moving = np.roll(big, (2, -3, 5), axis=(0, 1, 2))[2:18, 2:78, 2:78]
    best, dz, dy, dx = shift_search(ref, moving, 2, 8)
    assert best > 0.99
    assert (dz, dy, dx) == (2, -3, 5)


def test_check_pair_reports_offset_and_mismatch(tmp_path):
    vol = _texture((48, 384, 384))
    ref = _write_ome(tmp_path / "ref.zarr", vol, 2.0)
    # The same surface, shifted 20 voxels (40 um) in v: same sheet, offset.
    shifted = _write_ome(tmp_path / "shifted.zarr", np.roll(vol, 20, axis=1), 2.0)
    # A different surface: no placement explains it.
    other = _write_ome(tmp_path / "other.zarr", _texture((48, 384, 384), seed=1), 2.0)
    windows = [(150.0, 150.0), (250.0, 200.0)]

    same = check_pair(ref, ref, size_um=400, search_um=64, windows=windows)
    assert same["verdict"] == "placed" and same["median_ncc0"] > 0.99

    off = check_pair(ref, shifted, size_um=400, search_um=64, windows=windows)
    assert off["verdict"] == "same sheet, offset"
    assert off["median_best"] > 0.9
    assert all(w["shift_um"][1] == 40.0 for w in off["windows"])

    bad = check_pair(ref, other, size_um=400, search_um=64, windows=windows)
    assert bad["verdict"] == "no match within the search range"
