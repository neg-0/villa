"""Score ink predictions made at several depth offsets of one segment, to find whether its mesh sits off the inked layer.

Run the ink model once per layer window (for example optimized_inference with START_LAYER/END_LAYER shifted by a few
layers each time), then pass each prediction with its offset in layers:

python score_depth_sweep.py --map -8 pred_m8.png --map 0 pred_0.png --map 8 pred_p8.png [--labels inklabels.png]

For every offset it reports the fraction of pixels above 0.5 and the mean probability, which need no labels. The
label-free pick is the offset with the largest fraction above 0.5. With --labels it also reports the AUC per offset
and the label-best offset. Labels may be larger than the maps: --label-origin gives the map's top-left corner in
label pixels and --label-search searches a translation around it, keeping the one whose labels correlate best with
the mean of the maps; every offset is scored at that one translation.

The label-free pick is a diagnostic, not a correction: on 8 labelled PHerc0139 segments (villa #1912) it raised mean
AUC from 0.756 to 0.817 but lost more than 0.02 on two segments whose meshes were already aligned.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_map(path, window=None) -> np.ndarray:
    """A 2D map scaled to 0-1. window=(y0, x0, y1, x1) reads only that part (OME-Zarr / zarr labels: level 0)."""
    p = Path(path)
    is_zarr = str(path).rstrip("/").endswith(".zarr")
    if is_zarr:
        import zarr

        g = zarr.open(str(path), mode="r")
        a = g["0"] if hasattr(g, "keys") and "0" in g else g
        if a.ndim == 3:  # (1, h, w) label volumes
            a = a[0] if a.shape[0] == 1 else a[a.shape[0] // 2]
        a = a[window[0]:window[2], window[1]:window[3]] if window else a[:]
    elif p.suffix == ".npy":
        a = np.load(p)
    elif p.suffix in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(p)
    else:
        a = np.asarray(Image.open(p))
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if window and not is_zarr:
        a = a[window[0]:window[2], window[1]:window[3]]
    return a / 255.0 if a.max() > 1.0 else a


def auc(score, pos, neg, max_samples=200000) -> float:
    sp, sn = score[pos], score[neg]
    if sp.size < 50 or sn.size < 50:
        return float("nan")
    rng = np.random.default_rng(0)
    sp = rng.choice(sp, min(sp.size, max_samples), replace=False)
    sn = rng.choice(sn, min(sn.size, max_samples), replace=False)
    ranks = np.concatenate([sp, sn]).argsort(kind="mergesort").argsort() + 1
    return float((ranks[: sp.size].sum() - sp.size * (sp.size + 1) / 2) / (sp.size * sn.size))


def _best_shift(pred, labels, origin, search, step):
    h, w = pred.shape
    a = pred - pred.mean()
    aa = (a * a).sum()
    best = (-2.0, origin[0], origin[1])
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            y, x = origin[0] + dy, origin[1] + dx
            if y < 0 or x < 0 or y + h > labels.shape[0] or x + w > labels.shape[1]:
                continue
            L = labels[y:y + h, x:x + w]
            if L.std() == 0:
                continue
            b = L - L.mean()
            r = float((a * b).sum() / np.sqrt(aa * (b * b).sum() + 1e-12))
            if r > best[0]:
                best = (r, y, x)
    return best


def _block_mean(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def align_labels(pred, labels, origin=(0, 0), search=0, step=2, coarse=8):
    """The labels window under pred, at the translation within +-search of origin that correlates best.

    Searches larger than 4 * coarse run on coarse-times downsampled maps first, then refine at full resolution
    within +-coarse of the coarse best; smaller ones search every step pixels at full resolution.
    """
    h, w = pred.shape
    if search > 4 * coarse and min(h, w) >= 4 * coarse:
        _, y, x = _best_shift(_block_mean(pred, coarse), _block_mean(labels, coarse),
                              (origin[0] // coarse, origin[1] // coarse), -(-search // coarse), 1)
        _, y, x = _best_shift(pred, labels, (y * coarse, x * coarse), coarse, 1)
    else:
        _, y, x = _best_shift(pred, labels, origin, search, step if search else 1)
    return labels[y:y + h, x:x + w], (y, x)


def score(maps: dict, labels=None, label_origin=(0, 0), label_search=0, neg_band_px=None) -> dict:
    """Per-offset stats, and with labels the AUC per offset.

    With label_search, one translation is found for the whole sweep (on the mean of the maps) and every offset is
    scored at it, so no offset gets its own fit. A label misregistration does not change with depth, and a separate
    search per map let weak maps fit noise (V-025: 0.50 unsearched read as 0.60 to 0.87). auc_at_origin is the
    unsearched AUC, for comparison with pipelines that do not search.
    """
    offsets = sorted(maps)
    rows = {o: {"frac_above_0.5": float((maps[o] > 0.5).mean()), "mean_prob": float(maps[o].mean())} for o in offsets}
    res = {"offsets": rows, "label_free_pick": max(offsets, key=lambda o: rows[o]["frac_above_0.5"])}
    if labels is not None:
        shape = maps[offsets[0]].shape
        _, at = align_labels(np.mean([maps[o] for o in offsets], axis=0), labels, label_origin, label_search)
        res["label_origin"] = [int(v) for v in at]
        for o in offsets:
            rows[o]["auc"] = auc_at(maps[o], labels, at, neg_band_px)
            if label_search:
                rows[o]["auc_at_origin"] = auc_at(maps[o], labels, label_origin, neg_band_px)
        scored = [o for o in offsets if np.isfinite(rows[o]["auc"])]
        if scored:
            res["label_best"] = max(scored, key=lambda o: rows[o]["auc"])
    return res


def auc_at(pred, labels, at, neg_band_px=None) -> float:
    """AUC of pred against the labels window whose top-left corner is at (y, x); nan if it runs off the labels."""
    y, x = at
    if y < 0 or x < 0:
        return float("nan")
    L = labels[y:y + pred.shape[0], x:x + pred.shape[1]]
    if L.shape != pred.shape:
        return float("nan")
    neg = L < 0.05
    if neg_band_px is not None:
        neg &= ndimage.distance_transform_edt(L < 0.5) <= neg_band_px
    return auc(pred, L >= 0.5, neg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", nargs=2, required=True, metavar=("OFFSET", "PATH"),
                    help="prediction (png, tif or npy, values 0-1 or 0-255) made at OFFSET layers from the mesh")
    ap.add_argument("--labels", help="ink labels (white = ink)")
    ap.add_argument("--label-origin", default="0,0", help="map's top-left corner in label pixels, as y,x")
    ap.add_argument("--label-search", type=int, default=0, help="translation search around --label-origin, in pixels")
    ap.add_argument("--neg-band-px", type=float, help="score only non-ink pixels within this distance of ink")
    ap.add_argument("--output", help="write the report as JSON")
    a = ap.parse_args(argv)
    maps = {}
    for off, path in a.map:
        maps[int(off)] = load_map(path)
    labels = load_map(a.labels) if a.labels else None
    origin = tuple(int(v) for v in a.label_origin.split(","))
    res = score(maps, labels, origin, a.label_search, a.neg_band_px)
    for o, r in res["offsets"].items():
        print(o, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("label-free pick:", res["label_free_pick"], "| label best:", res.get("label_best"))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
