# Quality checks used for the Oct 9 changes

Run from `conversion/` after `./setup.sh` (needs the EdgeTAM checkpoint):

```bash
cd eval
python compare_sizes.py --clip clip_selfie --box 314 555 455 766 --flops   # 1024 vs 768 vs 512 agreement + GFLOPs
python group_eval.py 0 1 2 3 4                                           # hit rate / centre error vs true face positions
python memory_eval.py longocc_geom.json clip_longocc                     # memory presets (2,6) vs (2,3) vs (1,3)
python inspect_face.py clip_crowd <x> <y> <w> <h> <tag>                  # worst frames 1024 vs 512, side by side
```

The clips are synthetic: real face photos moved, scaled and hidden with known geometry
(`*_geom.json`), so the true face position is known in every frame.
