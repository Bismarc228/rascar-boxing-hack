# Per-Video Color Identity Hypotheses

Context: current pose extraction assigns `red` / `blue` roles with global HSV
thresholds in `tools/run_pose_baseline.py`. `classify_person_features()` scores
torso, shorts, and glove ROIs against fixed red/blue/white masks; then
`BoxingRoleManager` combines color score with short center memory and writes
`score_red` / `score_blue` into JSONL tracks. `rascar_boxing/pose_heuristic.py`
uses those roles as fighter labels and uses `score_{role}` as a small punch
score multiplier plus diagnostics. New hypothesis: fighter identity cannot be
global color identity because equipment changes by video. Calibrate identity
inside each video before trusting color margins.

No GPU run and no Kaggle submit from this note.

## Practical Plan

1. Build a per-video color calibration pass over cached pose tracks plus source
   frames, not over global train statistics.
   Sample frames every `N=10..20` or only frames where both fighters are visible
   and bboxes/keypoints are stable. For each detected person track, extract the
   same pose-guided ROIs already used by the role manager: torso polygon,
   shorts polygon, and wrist/glove circles. Keep ROI metadata: frame, track_id,
   bbox size, keypoint confidence, occlusion proxy, and whether both roles were
   present.

2. Estimate two video-local identity prototypes.
   Instead of asking "is this globally red or blue?", cluster appearance within
   the video into two persistent fighter identities. Start with tracklets from
   `track_id`; merge/split with bbox IoU, center continuity, and keypoint
   similarity. For each tracklet, aggregate robust color descriptors from
   high-confidence ROIs, then solve a two-cluster assignment over tracklets. Use
   color as appearance evidence and motion continuity as the constraint.

3. Map local prototypes to dataset labels `red` / `blue`.
   The local clusters are only `fighter_A` and `fighter_B` until a mapping is
   chosen. On train/validation, choose the mapping that maximizes matched GT
   punch fighter labels after time matching. On test, infer mapping from the
   current role manager if its per-video margin is strong, from corner/side
   priors only if measured, or keep the old labels when calibration confidence
   is low. Do not assume glove/short color names imply label names.

4. Replace frame-level color argmax with smoothed identity evidence.
   After per-video prototypes exist, each frame's candidates should get
   likelihoods `P(tracklet belongs to A/B)` from ROI color distance plus temporal
   continuity. Emit calibrated `score_red` / `score_blue` only after applying the
   video-local A/B-to-red/blue mapping. Low-confidence frames should inherit
   identity from the tracklet rather than flip on a single noisy crop.

## Crop Features

- ROI coverage and quality: masked pixel count, bbox-relative ROI area, mean
  keypoint confidence, visible limb count, saturation/value percentiles, and
  blur/edge magnitude to downweight motion blur.
- Color histograms: HSV hue histogram on saturated pixels, LAB `a/b`
  histogram, normalized RGB/chromaticity histogram, and simple red/blue/white/
  black fractions for compatibility with the current code.
- Prototype distances: cosine/chi-square distance to per-video fighter
  histograms by ROI type, margin between best and second prototype, and temporal
  EWMA of that margin.
- ROI-specific descriptors: glove color is high leverage when visible but often
  blurred/occluded; shorts and waistband are usually more stable; torso can be
  contaminated by referee/ropes/skin. Keep separate glove/shorts/torso scores
  and learn or grid-search weights instead of averaging them blindly.
- Track-level aggregates: median and trimmed-mean color features per tracklet,
  p10/p90 ranges, number of valid ROI observations, consecutive-frame identity
  switch count, and agreement between glove/shorts/torso evidence.

## Validation Against GT Fighter Labels

- Use train/validation videos only. Generate candidate punch rows from existing
  cached tracks, then match to GT with the repo metric's temporal assignment.
  Compare fighter accuracy before/after per-video remapping on matched events.
- Report both full macro and identity-specific diagnostics: `score_fighter`,
  wrong-fighter near misses inside `+/-15` frames, red/blue confusion matrix,
  per-video wins/losses, and deltas by tournament root.
- Add an ablation table:
  `old_global_hsv`, `per_video_gloves`, `per_video_shorts`, `per_video_all_rois`,
  `per_video_all_rois+track_smoothing`, and `oracle_train_mapping`.
- Use an oracle upper-bound on validation: choose the A/B-to-red/blue mapping
  per video using GT labels. If oracle mapping barely improves fighter score,
  color identity is not the bottleneck. If oracle improves but unsupervised test
  mapping is unstable, the bottleneck is label mapping rather than crop features.
- Keep timing/count fixed when auditing identity. Start from the same selected
  pose candidates and only swap fighter labels, so any score movement is
  attributable to identity.

## Failure Modes

- Equipment color does not correspond to official `red` / `blue` label in a
  video, or both fighters wear similar glove/short colors.
- Gloves are tiny, motion-blurred, hidden by guard/contact, or outside the
  pose-estimated wrist circle during punches.
- Shorts/torso ROIs include opponent limbs, ropes, referee, audience, skin, or
  canvas because keypoints are wrong or the fighter is bent/clinching.
- YOLO `track_id` switches split one fighter into many tracklets; color
  clustering can then treat the same fighter under different lighting as two
  identities.
- Lighting, camera white balance, compression, and venue color cast shift HSV
  thresholds; this is exactly why global red/blue masks should become auxiliary
  features, not identity rules.
- Mirrored camera angle or round transitions can invert left/right/corner
  assumptions. Side/corner priors must be validated per video, not hard-coded.
- Referee white filtering can remove valid light clothing or miss black/dark
  clothing; the current `ref > 0.15 and red < 0.1 and blue < 0.1` heuristic is
  too global for per-video apparel.

## First CPU-Only Experiment

Create an evaluator that reads one cached validation tracks directory, source
train videos, and GT punches. For each video, extract ROI descriptors on sampled
frames, cluster tracklets into two identities, compute an oracle GT mapping and
an unsupervised mapping, then rescore only fighter labels on the current
selected candidates. Gate any implementation on: higher `score_fighter`, no
macro regression from fixed timing/count, fewer role `track_id` switches, and a
clear per-video confidence signal that can be used safely on test.
