"""Find out why ink detection fails on one segment of a new scan, before reading it.

Open problem 10 lists three causes for a model that does not read a segment: the surface is in the wrong place, the
mesh sits off the inked layer, or the model does not transfer to the scan. This runs one check for each and writes
one report per segment:

1. Placement: compares the segment's surface volume with the same segment rendered on a reference scan
   (foundation/volume-registration/check_placement.py). Skipped without --reference.
2. Depth: runs the ink model on 62-layer windows shifted by --offsets layers and scores them together
   (ink-detection/depth_sweep/score_depth_sweep.py).
3. Model: with --labels, the AUC of the canonical model at the best offset. With --adapted-checkpoint (from
   ink-detection/scan_adapt/finetune_on_scan.py) it also runs the adapted model at that offset and reports the gain.

python diagnose_segment.py --surface-volume seg_newscan.zarr --checkpoint canonical.ckpt \
  --reference seg_refscan.zarr --crop 2000,3000,1024,1024 --labels inklabels.png --label-origin 2000,3000 \
  --label-search 32 --out-dir report/

--surface-volume is an 8-bit OME-Zarr surface volume (local path or URL; level 0 is used) or a (layers, height,
width) .npy; the placement check needs OME-Zarr for both it and --reference. --crop y,x,h,w limits inference to a
window of it in pixels. Labels are in the surface volume's pixels; only the part around the crop is read, and a
--label-search above 32 px runs coarse to fine (8x downsampled first), so a few hundred pixels takes seconds.
Writes out_dir/report.json, out_dir/report.md and one prediction PNG per offset and model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
for sub in ("foundation/volume-registration", "ink-detection/depth_sweep"):
    sys.path.insert(0, str(HERE.parent.parent / sub))

from check_placement import check_pair  # noqa: E402
from score_depth_sweep import load_map, score  # noqa: E402

LAYERS = 62


def load_stack(path, crop=None, z_range=None) -> np.ndarray:
    """(height, width, layers) uint8 from an OME-Zarr surface volume or a (layers, h, w) .npy, cropped."""
    if str(path).endswith(".npy"):
        a = np.load(path, mmap_mode="r")
    else:
        import zarr

        g = zarr.open(str(path), mode="r")
        a = g["0"] if hasattr(g, "keys") and "0" in g else g
    z0, z1 = z_range if z_range else (0, a.shape[0])
    y, x, h, w = crop if crop else (0, 0, a.shape[1], a.shape[2])
    s = np.asarray(a[z0:z1, y:y + h, x:x + w])
    if s.dtype != np.uint8:
        raise ValueError(f"{path}: expected 8-bit layers like the ink models' training data, got {s.dtype}")
    return np.ascontiguousarray(np.moveaxis(s, 0, -1))


def window(n_layers, offset):
    """[start, end) of the 62-layer window centred offset layers from the middle of n_layers."""
    start = n_layers // 2 - LAYERS // 2 + offset
    if start < 0 or start + LAYERS > n_layers:
        raise ValueError(f"offset {offset} needs layers [{start}, {start + LAYERS}) but the stack has {n_layers}")
    return start, start + LAYERS


def make_predictor(checkpoint, device="cuda", tile=256, stride=64, batch=4, work_dir="/tmp/diagnose_segment"):
    """predict(stack_hwc, start, end) -> (h, w) float32 probabilities, using optimized_inference's pipeline."""
    sys.path.insert(0, str(HERE.parent / "optimized_inference"))
    import torch
    import zarr
    from inference import CFG, run_inference
    from model_resnet3d_3d_decoder import load_model

    dev = torch.device(device)
    model = load_model(checkpoint, dev, num_frames=LAYERS)
    CFG.in_chans, CFG.tile_size, CFG.size, CFG.stride, CFG.batch_size = LAYERS, tile, tile, stride, batch
    CFG.workers, CFG.num_parts, CFG.part_id, CFG.zarr_output_dir = 0, 1, 0, str(work_dir)

    def predict(stack, start, end):
        res = run_inference(stack, model, dev, start_z=start, end_z=end)
        pred, cnt = zarr.open(res["mask_pred"], mode="r")[:], zarr.open(res["mask_count"], mode="r")[:]
        return np.where(cnt > 0, pred / np.maximum(cnt, 1e-6), 0).astype(np.float32)

    return predict


def load_labels(path, origin, shape, search):
    """Labels around the crop: only origin +- search plus the crop is read. Returns (labels, origin in them)."""
    if not path:
        return None, origin
    y0, x0 = max(0, origin[0] - search), max(0, origin[1] - search)
    labels = load_map(path, (y0, x0, origin[0] + shape[0] + search, origin[1] + shape[1] + search))
    return labels, (origin[0] - y0, origin[1] - x0)


def diagnose(stack, predict, offsets, labels=None, label_origin=(0, 0), label_search=0, neg_band_px=None,
             reference=None, surface_volume=None, adapted_predict=None, auc_ok=0.75, out_dir=None,
             placement_opts=None) -> dict:
    out = Path(out_dir) if out_dir else None
    rep = {"offsets": list(offsets)}
    if reference:
        rep["placement"] = {k: v for k, v in check_pair(reference, surface_volume, **(placement_opts or {})).items() if k != "windows"}
    maps = {}
    for o in offsets:
        maps[o] = predict(stack, *window(stack.shape[2], o))
        if out:
            Image.fromarray((np.clip(maps[o], 0, 1) * 255).astype(np.uint8)).save(out / f"pred_canonical_o{o:+d}.png")
    rep["depth"] = score(maps, labels, label_origin, label_search, neg_band_px)
    best = rep["depth"].get("label_best", rep["depth"]["label_free_pick"])
    rep["model"] = {"offset": best}
    if labels is not None:
        rep["model"]["canonical_auc"] = rep["depth"]["offsets"][best].get("auc")
    if adapted_predict is not None:
        m = adapted_predict(stack, *window(stack.shape[2], best))
        if out:
            Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(out / f"pred_adapted_o{best:+d}.png")
        rep["model"]["adapted_frac_above_0.5"] = float((m > 0.5).mean())
        if labels is not None:
            one = score({best: m}, labels, label_origin, label_search, neg_band_px)["offsets"][best]
            rep["model"]["adapted_auc"] = one.get("auc")
    rep["findings"] = findings(rep, auc_ok)
    return rep


def findings(rep, auc_ok=0.75) -> list:
    """Plain-language reading of the three checks, in the order they should be fixed."""
    f = []
    pl = rep.get("placement")
    if pl:
        v = pl["verdict"]
        if v.startswith("no match"):
            f.append("Placement: the surface does not match the reference scan within the search range. Fix the "
                     "transform or mesh first; the depth and model results below are not meaningful until then.")
        elif v == "same sheet, offset":
            f.append(f"Placement: same sheet as the reference, offset {pl['median_inplane_shift_um']:.0f} um in-plane. "
                     "Labels carried over from the reference need that shift.")
        elif v == "placed":
            f.append("Placement: matches the reference scan.")
        else:
            f.append(f"Placement: {v}.")
    d = rep["depth"]
    rows, pick = d["offsets"], d["label_free_pick"]
    if "label_best" in d:
        lb, a0 = d["label_best"], rows.get(0, {}).get("auc")
        gain = rows[lb]["auc"] - a0 if a0 is not None and np.isfinite(a0) else None
        if lb != 0 and gain is not None and gain >= 0.02:
            f.append(f"Depth: ink reads best {lb:+d} layers from the mesh (AUC {a0:.3f} at 0, {rows[lb]['auc']:.3f} "
                     f"at {lb:+d}). The mesh sits off the inked layer.")
        else:
            f.append(f"Depth: the mesh is on the inked layer (best offset {lb:+d}).")
    elif pick != 0:
        f.append(f"Depth: without labels, the most ink is predicted {pick:+d} layers from the mesh. Check that offset "
                 "by eye before trusting it; on aligned meshes this pick can be worse than 0.")
    else:
        f.append("Depth: without labels, the most ink is predicted at the mesh (offset 0).")
    m = rep["model"]
    a = m.get("canonical_auc")
    if a is not None and np.isfinite(a):
        if a < auc_ok:
            f.append(f"Model: the canonical model reaches only AUC {a:.3f} at the best offset. If placement is fine, "
                     "the model does not transfer to this scan; adapt it with ink-detection/scan_adapt.")
        else:
            f.append(f"Model: the canonical model reads this scan (AUC {a:.3f} at the best offset).")
    if "adapted_auc" in m and a is not None:
        f.append(f"Model: the adapted model gives AUC {m['adapted_auc']:.3f} ({m['adapted_auc'] - a:+.3f}).")
    elif "adapted_frac_above_0.5" in m:
        f.append(f"Model: the adapted model marks {m['adapted_frac_above_0.5']:.1%} of pixels as ink "
                 f"(canonical: {rows[m['offset']]['frac_above_0.5']:.1%}). Without labels this is not a score.")
    return f


def write_markdown(rep, path):
    lines = ["# Segment diagnosis", ""] + [f"- {x}" for x in rep["findings"]] + ["", "| offset | frac > 0.5 | AUC |",
                                                                                "| --- | --- | --- |"]
    for o, r in rep["depth"]["offsets"].items():
        lines.append(f"| {o:+d} | {r['frac_above_0.5']:.3f} | {r.get('auc', float('nan')):.3f} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface-volume", required=True, help="OME-Zarr surface volume or (layers, h, w) .npy")
    ap.add_argument("--checkpoint", required=True, help="canonical ResNet3D-152 3D-decoder checkpoint")
    ap.add_argument("--reference", help="the same segment's surface volume on a reference scan (OME-Zarr)")
    ap.add_argument("--adapted-checkpoint", help="model adapted to this scan by scan_adapt/finetune_on_scan.py")
    ap.add_argument("--offsets", default="-16,-8,-4,0,4,8,16", help="window offsets in layers")
    ap.add_argument("--crop", help="y,x,h,w window of the surface volume in pixels")
    ap.add_argument("--labels", help="ink labels (white = ink; png, tif, npy or zarr, level 0), in surface-volume pixels")
    ap.add_argument("--label-origin", default="0,0", help="the crop's top-left corner in label pixels, as y,x")
    ap.add_argument("--label-search", type=int, default=0)
    ap.add_argument("--neg-band-px", type=float, help="score only non-ink pixels within this distance of ink")
    ap.add_argument("--auc-ok", type=float, default=0.75, help="canonical AUC below this suggests adapting the model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    offsets = [int(v) for v in a.offsets.split(",")]
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    stack = load_stack(a.surface_volume, crop)
    predict = make_predictor(a.checkpoint, a.device, work_dir=out / "partitions")
    adapted = make_predictor(a.adapted_checkpoint, a.device, work_dir=out / "partitions") if a.adapted_checkpoint else None
    labels, origin = load_labels(a.labels, tuple(int(v) for v in a.label_origin.split(",")), stack.shape[:2],
                                 a.label_search)
    rep = diagnose(stack, predict, offsets, labels, origin, a.label_search,
                   a.neg_band_px, a.reference, a.surface_volume, adapted, a.auc_ok, out)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    write_markdown(rep, out / "report.md")
    print("\n".join(rep["findings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
