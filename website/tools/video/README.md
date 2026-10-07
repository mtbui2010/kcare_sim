# Tutorial video sources

Regenerate with the narrated-tutorial-video skill (voice env: `bash setup_env.sh .video-env`):

```bash
cd website/tools/video
$ENV/bin/python $SKILL/scripts/tts.py narration.json voice --voice am_michael --speed 1.15
$ENV/bin/python render_video.py storyboard.json --out tutorial.mp4    # local copy of the skill renderer + layout "side"
$ENV/bin/python $SKILL/scripts/check_video.py tutorial.mp4 --narration narration.json --storyboard storyboard.json --sheet sheet.png
cp tutorial.mp4 ../../docs/assets/video/
```

- `tour_clip.mp4`, `vla_clip.mp4` + `*_overlay.json`: recorded with `sim/scripts/house_record_grid.py` through
  kcare's HTTP API on 7 Oct 2026. The VLA run used checkpoint officehouse_20261007/005000; the drawer opened
  28.6 cm in 18.7 s.
- `out/*.txt`: outputs captured from the real runs (see `website/tools/data/out/`).
- `plan_vs_sim.png`, `calib_home.jpg`, `depth_compare.jpg`: from the sim (`house_scene.py --render`,
  `calibrate_home.py`, `depth_compare.py`).
