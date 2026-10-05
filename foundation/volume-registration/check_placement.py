"""Check that one segment's surface volumes on two scans of the same scroll show the same papyrus.

A segment rendered on two scans shares one (u, v) parameterisation, which each surface volume's OME-Zarr scale
maps to micrometres. If both meshes sit on the same sheet, the two renders show the same structure at the same
(u, v, depth). For random windows inside the reference's data this reports, on a common grid of twice the coarser
voxel size:
  ncc0   normalised cross-correlation of a central depth slab with no shift
  best   the highest NCC over depth shifts (+-depth_layers grid layers) and in-plane shifts (+-search_um)
  shift  the (depth, v, u) shift in um that gives `best`

A pair whose best NCC is low has no placement within the search range that explains the other render (for example
a wrong transform). A pair with a high best NCC but a large shift is placed on the right sheet, but offset.

Example:
python check_placement.py \
--reference https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/<seg>/surface-volumes/2.399um-...zarr \
--other https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/<seg>/surface-volumes/1.129um-...zarr
"""

import argparse
import json
import sys
from typing import Optional

import numpy as np
import zarr
from scipy import ndimage
from scipy.signal import fftconvolve

NO_MATCH_NCC = 0.4


def ome_scales(group) -> list:
    """Per-level (z, y, x) voxel size in um from OME-Zarr multiscales metadata."""
    datasets = group.attrs["multiscales"][0]["datasets"]
    return [d["coordinateTransformations"][0]["scale"] for d in datasets]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def shift_search(ref: np.ndarray, moving: np.ndarray, dz: int, m: int):
    """Highest NCC of `ref` over every placement inside `moving`, which is `ref` padded by dz in depth and m in y/x.

    Returns (ncc, dz, dy, dx) with shifts in grid cells.
    """
    r = ref - ref.mean()
    rn = np.sqrt((r * r).sum())
    k = np.ones(ref.shape, np.float32)
    num = fftconvolve(moving, r[::-1, ::-1, ::-1], mode="valid")
    s1 = fftconvolve(moving, k, mode="valid")
    s2 = fftconvolve(moving * moving, k, mode="valid")
    c = num / (rn * np.sqrt(np.maximum(s2 - s1 * s1 / k.size, 1e-6)))
    i = np.unravel_index(np.argmax(c), c.shape)
    return float(c[i]), int(i[0] - dz), int(i[1] - m), int(i[2] - m)


def pick_level(scales: list, grid_um: float) -> int:
    levels = [i for i, s in enumerate(scales) if s[1] <= grid_um * 1.05]
    return levels[-1] if levels else 0


def read_padded(arr, start, stop) -> np.ndarray:
    """Read arr[start:stop] with zeros outside the array."""
    out = np.zeros([b - a for a, b in zip(start, stop)], np.float32)
    src = tuple(slice(max(a, 0), min(b, n)) for a, b, n in zip(start, stop, arr.shape))
    if all(s.stop > s.start for s in src):
        dst = tuple(slice(s.start - a, s.stop - a) for s, a in zip(src, start))
        out[dst] = arr[src]
    return out


def read_um(group, scales, grid_um, y0_um, x0_um, size_um, nlayers) -> np.ndarray:
    """A size_um square at (y0_um, x0_um) with nlayers grid layers centred in depth, resampled to grid_um."""
    lv = pick_level(scales, grid_um)
    s = scales[lv]
    arr = group[str(lv)]
    zc = arr.shape[0] // 2
    hz = int(np.ceil(nlayers * grid_um / s[0] / 2)) + 1
    y0, x0, n = int(y0_um / s[1]), int(x0_um / s[2]), int(np.ceil(size_um / s[1]))
    hz = min(hz, zc, arr.shape[0] - zc)  # surface volumes are thin: keep the layers that exist
    a = read_padded(arr, [zc - hz, y0, x0], [zc + hz, y0 + n, x0 + n])
    a = ndimage.zoom(a, (s[0] / grid_um, s[1] / grid_um, s[2] / grid_um), order=1)
    c, h = a.shape[0] // 2, min(nlayers, a.shape[0]) // 2
    return a[c - h:c + h]


def sample_windows(group, scales, n: int, size_um: float, seed: int = 0) -> list:
    """Up to n window corners (um) whose whole square lies inside the data at the coarsest level."""
    lv = len(scales) - 1
    arr = group[str(lv)]
    mid = np.asarray(arr[arr.shape[0] // 2]) > 0
    ok = ndimage.binary_erosion(mid, iterations=max(1, int(np.ceil(size_um / scales[lv][1]))))
    ys, xs = np.nonzero(ok)
    if len(ys) == 0:
        return []
    idx = np.random.default_rng(seed).choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(float(ys[i] * scales[lv][1] - size_um / 2), float(xs[i] * scales[lv][2] - size_um / 2)) for i in idx]


def check_pair(
    reference: str,
    other: str,
    n_windows: int = 6,
    size_um: float = 1500.0,
    search_um: float = 200.0,
    depth_layers: int = 8,
    half_slab: int = 12,
    windows: Optional[list] = None,
) -> dict:
    ref_g, oth_g = zarr.open(reference, mode="r"), zarr.open(other, mode="r")
    ref_s, oth_s = ome_scales(ref_g), ome_scales(oth_g)
    grid = 2 * max(ref_s[0][1], oth_s[0][1])
    m = int(np.ceil(search_um / grid))
    if windows is None:
        windows = sample_windows(ref_g, ref_s, n_windows, size_um)
    out = []
    for y_um, x_um in windows:
        w = {"window_um": [round(y_um), round(x_um)]}
        out.append(w)
        r = read_um(ref_g, ref_s, grid, y_um, x_um, size_um, 2 * half_slab)
        x = read_um(oth_g, oth_s, grid, y_um, x_um, size_um, 2 * half_slab + 2 * depth_layers)
        n = min(r.shape[1], r.shape[2], x.shape[1], x.shape[2])
        hs = min(half_slab, r.shape[0] // 2, x.shape[0] // 2 - 2)
        dz = min(depth_layers, x.shape[0] // 2 - hs)
        if hs < 3 or dz < 0 or n <= 2 * m + 8:
            w["skipped"] = "too few layers or pixels"
            continue
        zr, zx = r.shape[0] // 2, x.shape[0] // 2
        r = r[zr - hs:zr + hs, :n, :n][:, m:n - m, m:n - m]
        x = x[zx - hs - dz:zx + hs + dz, :n, :n]
        if (r > 0).mean() < 0.9 or (x > 0).mean() < 0.9:
            w["skipped"] = "under 90% data"
            continue
        best, sz, sy, sx = shift_search(r, x, dz, m)
        w.update(
            ncc0=round(ncc(r, x[dz:dz + 2 * hs, m:n - m, m:n - m]), 3),
            best=round(best, 3),
            shift_um=[round(t * grid, 1) for t in (sz, sy, sx)],
        )
    good = [w for w in out if "best" in w]
    res = {"reference": reference, "other": other, "grid_um": round(grid, 3), "windows": out, "n": len(good)}
    if good:
        res["median_ncc0"] = float(np.median([w["ncc0"] for w in good]))
        res["median_best"] = float(np.median([w["best"] for w in good]))
        res["median_inplane_shift_um"] = float(np.median([np.hypot(*w["shift_um"][1:]) for w in good]))
        if res["median_best"] < NO_MATCH_NCC:
            res["verdict"] = "no match within the search range"
        elif res["median_inplane_shift_um"] > grid:
            res["verdict"] = "same sheet, offset"
        else:
            res["verdict"] = "placed"
    else:
        res["verdict"] = "not enough data"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="surface volume (OME-Zarr) to compare against")
    ap.add_argument("--other", required=True, action="append", help="surface volume of the same segment on another scan")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--size-um", type=float, default=1500.0)
    ap.add_argument("--search-um", type=float, default=200.0, help="in-plane search range")
    ap.add_argument("--depth-layers", type=int, default=8, help="depth search range in grid layers")
    ap.add_argument("--output", help="write the full per-window report as JSON")
    a = ap.parse_args(argv)
    reports = [
        check_pair(a.reference, o, a.windows, a.size_um, a.search_um, a.depth_layers) for o in a.other
    ]
    for r in reports:
        summary = {k: v for k, v in r.items() if k not in ("windows", "reference")}
        print(json.dumps(summary))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(reports, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
