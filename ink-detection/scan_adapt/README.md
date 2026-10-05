# Adapting the ink model to a new scan

`finetune_on_scan.py` fine-tunes the ResNet3D-152 3D-decoder ink model on a new scan of a scroll, with ink labels
carried over from a scan that already has them. Check first that the meshes for the new scan sit on the labelled
papyrus (`foundation/volume-registration/check_placement.py`); labels on a misplaced mesh teach the model the wrong
thing.

```bash
python finetune_on_scan.py --checkpoint canonical.ckpt --out-dir adapted/ \
  --pair w013_1129um.npy w013_inklabels.png --pair w018_1129um.npy w018_inklabels.png
```

Each stack is a `(layers, height, width)` uint8 surface volume, at least 62 layers deep, with labels of the same
height and width. The output `adapted/final.pt` loads with `optimized_inference`'s `load_model`.

Measured on PHerc1667 (2.4 um model adapted to the 1.129 um scan, 6 labelled segments, leave-two-out, 3 folds):
- Mean held-out AUC 0.564 zero-shot, 0.86 after 2000 iterations; all 6 segments improved.
- With the same recipe, meshes from the corrected transform (#1843) read held-out ink better than the published
  meshes on 6 of 6 segments (mean AUC 0.859 vs 0.651).
- bfloat16 autocast (the default) trained every fold with no skipped steps; float16 diverged on 2 of 3 folds.
- The adapted model drops on the original 2.399 um scan (0.92 to about 0.75), so keep one model per scan.

The training step is the one used for those runs. `--seed` sets the tile order; those runs used seed 0 + fold
index, so `--seed 1` reproduces fold 1's tiles. Tests (CPU, a stand-in model):
`python -m pytest test_finetune_on_scan.py`. Needs torch, numpy, scipy, pillow.
