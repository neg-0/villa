# Segment diagnosis

Runs the three checks for "why does ink fail on this segment of a new scan" (open problem 10) and writes one report:

1. **Placement**: the segment's surface volume against the same segment on a reference scan
   (`foundation/volume-registration/check_placement.py`).
2. **Depth**: the ink model on 62-layer windows shifted by `--offsets`, scored together
   (`ink-detection/depth_sweep/score_depth_sweep.py`).
3. **Model**: with `--labels`, the canonical model's AUC at the best offset; with `--adapted-checkpoint` (from
   `ink-detection/scan_adapt/finetune_on_scan.py`), the adapted model's score at the same offset.

```bash
python diagnose_segment.py --surface-volume seg_newscan.zarr --checkpoint canonical.ckpt \
  --reference seg_refscan.zarr --crop 2000,3000,1024,1024 \
  --labels inklabels.png --label-origin 2000,3000 --label-search 32 --out-dir report/
```

Writes `report/report.json`, `report/report.md` and one prediction PNG per offset. The findings are listed in the
order to fix them: a placement that matches nothing makes the depth and model results meaningless, and a canonical
AUC below `--auc-ok` (default 0.75) on a well-placed segment points at adapting the model.

Inference goes through `optimized_inference` (`run_inference`, tile 256, stride 64), so it needs that folder's
requirements and a GPU for full segments; use `--crop` to keep runs short. The tests need only numpy, scipy, pillow
and zarr: `python -m pytest test_diagnose_segment.py`.
