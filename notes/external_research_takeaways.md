# External Research Takeaways

Sources: `test.md` and `Boxing_Action_Recognition_чистый_стек_под_одну_камеру_+_видео.pdf`
from the local attachments.

## Useful Direction

- Optimize impact spotting first. Time is 50% of the metric, and the current
  pose heuristics still move mainly through better event timing.
- Treat `clear=false` as precision control: selected low-confidence events are
  just thresholded away in our generator, but a learned spotter should expose a
  calibrated confidence so the postprocess can choose precision over recall.
- The RGB-only single-camera stack recommendation matches our current best
  path: detector/pose/tracker, per-video fighter identity, then an event
  spotter and small clip classifier.
- T-DEED/E2E-Spot/Dense Detection Anchors are relevant if we build a real
  per-frame spotter. A practical local approximation is a pose/RGB temporal
  model with Gaussian labels and soft-NMS, then compare against current
  `yolo11s`/`yolo26l` peak selectors.
- Fighter identity should use video-local equipment prototypes. Fixed global
  red/blue HSV masks are only weak features; tracklet continuity and per-video
  color clustering matter more.
- Postprocess should keep a same-fighter refractory period around 6-10 frames
  while allowing near-simultaneous different-fighter events. This supports the
  existing `fighter_hand` grouped NMS and cross-NMS sweeps.

## Near-Term Experiments

1. Build a frame-level or candidate-level spotter with Gaussian labels around
   GT frames, using cached pose-derived features first; only move to RGB clips
   if it beats the simple pose peak heuristic on tournament roots.
2. Add per-video/tracklet color prototypes from central bbox or pose-guided
   torso/shorts/glove crops, then audit only fighter-label changes with fixed
   timing/counts.
3. Keep large pose models as detector witnesses. Use them for context rescoring,
   timing snap, and agreement, not blindly as replacements unless tournament
   roots and FP pass.
4. Delay punch-type/effectiveness modeling until selection/timing is stronger.
   Their metric weight is lower and current attribute priors are not the main
   bottleneck.
