"""Score ink predictions made at several depth offsets of one segment, to find whether its mesh sits off the inked layer.

Run the ink model once per layer window (for example optimized_inference with START_LAYER/END_LAYER shifted by a few
layers each time), then pass each prediction with its offset in layers:

python score_depth_sweep.py --map -8 pred_m8.png --map 0 pred_0.png --map 8 pred_p8.png [--labels inklabels.png]

For every offset it reports the fraction of pixels above 0.5 and the mean probability, which need no labels. The
label-free pick is the offset with the largest fraction above 0.5. With --labels it also reports the AUC per offset
and the label-best offset. Labels may be larger than the maps: --label-origin gives the map's top-left corner in
label pixels and --label-search searches a translation around it, keeping the one whose labels correlate best with
the map.

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


def load_map(path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        a = np.load(p)
    elif p.suffix in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(p)
    else:
        a = np.asarray(Image.open(p))
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
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


def align_labels(pred, labels, origin=(0, 0), search=0, step=2):
    """The labels window under pred, at the translation within +-search of origin that correlates best."""
    h, w = pred.shape
    a = pred - pred.mean()
    best = (-2.0, origin[0], origin[1])
    for dy in range(-search, search + 1, step if search else 1):
        for dx in range(-search, search + 1, step if search else 1):
            y, x = origin[0] + dy, origin[1] + dx
            if y < 0 or x < 0 or y + h > labels.shape[0] or x + w > labels.shape[1]:
                continue
            L = labels[y:y + h, x:x + w]
            if L.std() == 0:
                continue
            b = L - L.mean()
            r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
            if r > best[0]:
                best = (r, y, x)
    _, y, x = best
    return labels[y:y + h, x:x + w], (y, x)


def score(maps: dict, labels=None, label_origin=(0, 0), label_search=0, neg_band_px=None) -> dict:
    offsets = sorted(maps)
    rows = {o: {"frac_above_0.5": float((maps[o] > 0.5).mean()), "mean_prob": float(maps[o].mean())} for o in offsets}
    res = {"offsets": rows, "label_free_pick": max(offsets, key=lambda o: rows[o]["frac_above_0.5"])}
    if labels is not None:
        for o in offsets:
            L, at = align_labels(maps[o], labels, label_origin, label_search)
            if L.shape != maps[o].shape:
                rows[o]["auc"] = float("nan")
                continue
            neg = L < 0.05
            if neg_band_px is not None:
                neg &= ndimage.distance_transform_edt(L < 0.5) <= neg_band_px
            rows[o]["auc"] = auc(maps[o], L >= 0.5, neg)
            rows[o]["label_origin"] = [int(v) for v in at]
        scored = [o for o in offsets if np.isfinite(rows[o]["auc"])]
        if scored:
            res["label_best"] = max(scored, key=lambda o: rows[o]["auc"])
    return res


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
