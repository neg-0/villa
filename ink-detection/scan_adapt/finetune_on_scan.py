"""Adapt the ResNet3D-152 3D-decoder ink model to a new scan of a scroll, using ink labels carried over from another scan.

Typical case: a scroll has a labelled scan (for example 2.4 um) and a new scan (for example 1.1 um). Render the
labelled segments on the new scan along a mesh placed on the same papyrus (check with
foundation/volume-registration/check_placement.py first), so the labels line up with the new renders. Then:

python finetune_on_scan.py --checkpoint canonical.ckpt --out-dir adapted/ \
  --pair seg1_newscan.npy seg1_inklabels.png --pair seg2_newscan.npy seg2_inklabels.png

Each stack is a (layers, height, width) uint8 .npy surface volume at least 62 layers deep (the centre 62 are used),
and each label image has the same height and width (white = ink). Training uses only non-ink pixels within 2 mm of
ink as negatives (--near-mm, at --voxel-um), like the usual ink evaluation.

Recipe (measured on PHerc1667, 2.4 um model adapted to the 1.129 um scan, 6 labelled segments, leave-two-out):
2000 iterations of 2 tiles of 256 px, AdamW lr 2e-5, weight decay 1e-4, 100-iteration warm-up then cosine,
BatchNorm statistics frozen, bfloat16 autocast. It lifted mean held-out AUC from 0.56 to 0.86, and stayed stable
on every fold (float16 autocast diverged on 2 of 3 folds of one arm). The adapted model loses accuracy on the
original scan (0.92 to about 0.75 there), so keep one model per scan.
Writes out_dir/final.pt (state_dict, loadable by optimized_inference's load_model) and out_dir/train_log.json;
reruns resume from the latest out_dir/iter*.pt.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

TILE, OUT_TILE, LAYERS = 256, 64, 62


def load_pair(stack_path, label_path, near_px):
    ct = np.load(stack_path, mmap_mode="r")
    c = ct.shape[0] // 2
    ct = np.asarray(ct[c - LAYERS // 2:c + LAYERS // 2])
    if ct.shape[0] != LAYERS:
        raise ValueError(f"{stack_path}: need at least {LAYERS} layers, got {ct.shape[0]}")
    lab = np.asarray(Image.open(label_path).convert("L")) >= 128
    h, w = min(lab.shape[0], ct.shape[1]), min(lab.shape[1], ct.shape[2])
    ct, lab = ct[:, :h, :w], lab[:h, :w]
    near = ndimage.distance_transform_edt(~lab) <= near_px
    has_data = (ct != 0).any(0)
    return np.ascontiguousarray(ct), lab.astype(np.float32), (near & has_data).astype(np.float32)


def sample(pairs, rng):
    """One 256 px tile with >= 2% ink and >= 50% supervised pixels; random flips/rot90 and mild contrast jitter."""
    while True:
        ct, lab, m = pairs[rng.integers(len(pairs))]
        if ct.shape[1] < TILE or ct.shape[2] < TILE:
            raise ValueError("every stack must be at least 256 x 256 px")
        y, x = rng.integers(0, ct.shape[1] - TILE + 1), rng.integers(0, ct.shape[2] - TILE + 1)
        L, M = lab[y:y + TILE, x:x + TILE], m[y:y + TILE, x:x + TILE]
        if M.mean() < 0.5 or L.mean() < 0.02:
            continue
        X = np.clip(ct[:, y:y + TILE, x:x + TILE].astype(np.float32), 0, 200) / 200.0
        k = rng.integers(4)
        X, L, M = np.rot90(X, k, (1, 2)), np.rot90(L, k), np.rot90(M, k)
        if rng.random() < 0.5:
            X, L, M = X[:, :, ::-1], L[:, ::-1], M[:, ::-1]
        g, b = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05)
        X = np.where(X > 0, np.clip(X * g + b, 0, 1), 0)
        f = TILE // OUT_TILE
        pool = lambda a: np.ascontiguousarray(a).reshape(OUT_TILE, f, OUT_TILE, f).mean((1, 3))
        return np.ascontiguousarray(X), pool(L), (pool(M) > 0.5).astype(np.float32)


def forward(net, x, torch):
    """net(x) for the RegressionModel in eval mode, with gradient checkpointing through the backbone stages."""
    bb = getattr(net, "backbone", None)
    if bb is None or not hasattr(bb, "layer4"):
        return net(x)
    from torch.utils.checkpoint import checkpoint_sequential

    if net.normalization is not None:
        x = net.normalization(x)
    x = bb.relu(bb.bn1(bb.conv1(x)))
    if not bb.no_max_pool:
        x = bb.maxpool(x)
    feats = []
    for layer in (bb.layer1, bb.layer2, bb.layer3, bb.layer4):
        x = checkpoint_sequential(layer, max(1, len(layer) // 3), x, use_reentrant=False)
        feats.append(x)
    return net.decoder(feats)


def finetune(net, pairs, out_dir, device, iters=2000, lr=2e-5, weight_decay=1e-4, micro_batches=2, amp="bf16",
             seed=0, log_every=25, save_every=200):
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    if final.exists():
        print("already trained:", final)
        return json.loads((out_dir / "train_log.json").read_text())
    torch.manual_seed(seed)
    net.eval()  # BatchNorm statistics frozen; no deep-supervision heads
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    warm = min(100, max(1, iters // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, iters) / iters)))
    use_amp = device.type == "cuda" and amp != "fp32"
    scaler = torch.amp.GradScaler(enabled=use_amp and amp == "fp16")
    adt = torch.bfloat16 if amp == "bf16" else torch.float16
    it0, rng, log, skipped = 0, np.random.default_rng(seed), [], 0
    ckpts = sorted(glob.glob(str(out_dir / "iter*.pt")))
    if ckpts:
        s = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); it0, log, skipped = s["iter"], s["log"], s.get("skipped", 0)
        rng.bit_generator.state = s["rng"]
        print("resumed at", it0, flush=True)
    t0 = time.time()
    for it in range(it0, iters):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(micro_batches):  # micro-batches of 1; BatchNorm is frozen, so this equals one larger batch
            X, Y, M = (torch.from_numpy(a[None, None].copy()).to(device) for a in sample(pairs, rng))
            with torch.autocast(device.type, dtype=adt, enabled=use_amp):
                logit = forward(net, X, torch)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y, reduction="none") * M).sum()
            loss = loss / M.sum().clamp(min=1)
            if not torch.isfinite(loss):  # drop a non-finite micro-batch
                skipped += 1
                continue
            scaler.scale(loss / micro_batches).backward()
            tot += loss.item() / micro_batches
        if not any(p.grad is not None for p in net.parameters()):
            log.append(float("nan")); sched.step()
            continue
        scaler.unscale_(opt)
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)):
            opt.zero_grad(set_to_none=True); skipped += 1  # never step on non-finite gradients
        scaler.step(opt); scaler.update(); sched.step()
        log.append(tot)
        if (it + 1) % log_every == 0:
            print(f"it {it + 1}/{iters} loss {np.mean(log[-log_every:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % save_every == 0 or it + 1 == iters:
            tmp = out_dir / f"iter{it + 1:06d}.pt"
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "iter": it + 1, "rng": rng.bit_generator.state, "log": log,
                        "skipped": skipped}, str(tmp) + ".part")
            os.replace(str(tmp) + ".part", tmp)
            for old in sorted(glob.glob(str(out_dir / "iter*.pt")))[:-1]:
                os.remove(old)
    if not all(torch.isfinite(t).all() for t in net.state_dict().values() if t.is_floating_point()):
        raise RuntimeError("non-finite weights after fine-tuning; try --amp fp32 or a lower --lr")
    torch.save({"state_dict": net.state_dict()}, str(final) + ".part")
    os.replace(str(final) + ".part", final)
    info = {"loss": log, "iters": iters, "micro_batches": micro_batches, "lr": lr, "amp": amp, "seed": seed,
            "skipped_nonfinite": skipped, "skipped_frac": skipped / max(1, iters * micro_batches)}
    (out_dir / "train_log.json").write_text(json.dumps(info))
    if info["skipped_frac"] > 0.01:
        print(f"warning: {skipped} non-finite micro-batches ({info['skipped_frac']:.1%}); the run may be unstable")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="ResNet3D-152 3D-decoder checkpoint to start from")
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("STACK_NPY", "LABELS_PNG"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voxel-um", type=float, default=2.399, help="label pixel size, for the 2 mm negative band")
    ap.add_argument("--near-mm", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "optimized_inference"))
    from model_resnet3d_3d_decoder import load_model

    device = torch.device(a.device)
    net = load_model(a.checkpoint, device).model
    pairs = [load_pair(s, l, a.near_mm * 1000 / a.voxel_um) for s, l in a.pair]
    finetune(net, pairs, a.out_dir, device, iters=a.iters, lr=a.lr, amp=a.amp, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
