# Depth sweep scoring

Checks whether a segment's mesh sits off the inked layer (villa #1912). Run the ink model once per layer window,
shifted a few layers each time (for example `optimized_inference` with `START_LAYER`/`END_LAYER` moved together),
then score the predictions together:

```bash
python score_depth_sweep.py --map -8 pred_m8.png --map 0 pred_0.png --map 8 pred_p8.png \
  --labels inklabels.png --label-search 48 --output report.json
```

Per offset it reports the fraction of pixels above 0.5 and the mean probability (no labels needed), and with
`--labels` the AUC. It prints the label-free pick (largest fraction above 0.5) and the label-best offset.

Checked on the 25-offset sweep (−48 to +48 layers) of 8 labelled PHerc0139 March-scan segments from #1912: it
reproduces the earlier analysis exactly. Mean AUC is 0.740 at offset 0 of the flattened render, 0.817 at the
label-free pick and 0.841 at the label-best offset. The label-free pick is a diagnostic, not a fix: it loses
0.044 on w041 and helps most where the mesh is clearly off (w044 0.658 → 0.886, w045 0.607 → 0.788).

Needs numpy, scipy, pillow (tifffile for .tif maps). Tests: `python -m pytest test_score_depth_sweep.py`.
