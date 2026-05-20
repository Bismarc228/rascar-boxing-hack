# YOLO26 Pose Hypotheses

Context: user-proposed "yolka 26" means Ultralytics YOLO26 pose. Current public
best is `0.13275` from `yolo11s-pose` fighter+hand grouped NMS. The best local
`yolo11s` validation score is `0.340888`. `yolo11m` is stronger offline
(`0.363325`) but worse publicly (`0.12958`), so larger detectors need a
transfer-risk gate, not just a higher validation score. Do not run GPU work for
this note, and never touch GPU 0.

## Evidence

- Local caches exist for `val_yolo11s_conf035`, `test_yolo11s_conf035`,
  `val_yolo11m_conf035`, and `test_yolo11m_conf035`. The validation caches have
  13 videos; the test caches have 9 videos.
- Local model files currently present: `models/yolo11n-pose.pt`,
  `models/yolo11s-pose.pt`, and `models/yolo11m-pose.pt`. No `yolo26*-pose.pt`
  weights are present yet, so first use will need a download or explicit model
  placement.
- Local Ultralytics package is `8.4.24`; `Model.track(...)` and
  `Model.predict(...)` accept `**kwargs`, but this repo's extractor does not
  expose an `--end2end` flag.
- Official Ultralytics docs list YOLO26-pose weights:
  `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`,
  `yolo26l-pose.pt`, and `yolo26x-pose.pt`.
- Official YOLO26-pose COCO keypoint metrics for `n/s/m/l/x` are
  `57.2/63.0/68.8/70.4/71.6` mAP50-95(e2e). The requested practical scope is
  `n/s/m/l`; skip `x` until `l` gives a clear reason.
- YOLO26-pose keeps the COCO 17-keypoint layout, so the current wrist, elbow,
  shoulder, head, and body indexing should remain compatible if Ultralytics
  returns the same `results[0].keypoints.data` structure.

Official references:

- https://docs.ultralytics.com/models/yolo26/
- https://docs.ultralytics.com/tasks/pose/
- https://docs.ultralytics.com/guides/end2end-detection

## Weight Order

1. `yolo26n-pose.pt`: compatibility and tracker smoke only. It is the cheapest
   way to verify weight loading, `track()` behavior, JSONL schema, line counts,
   and role assignment. Do not expect it to beat `yolo11s`.
2. `yolo26s-pose.pt`: first real validation candidate. It is the closest
   public-risk analog to the successful `yolo11s` path and should be tested
   before any bigger model.
3. `yolo26m-pose.pt`: run only after `s` has sane validation behavior. Compare
   against both `yolo11s` and the failed-public `yolo11m` pattern; require a
   better transfer story than "bigger local score".
4. `yolo26l-pose.pt`: run only if `m` passes gates or if `s` is strong but
   recall-limited. Use `jobs=1`; this is likely expensive and most exposed to
   validation overfit.

## End2End And NMS Details

- YOLO26 defaults to the one-to-one end-to-end head. Official docs say this path
  is NMS-free and returns final detections directly; for pose the end-to-end
  output shape is `(N, 300, 57)`.
- The traditional one-to-many head can be selected with `end2end=False`; that
  path requires detector NMS and may be useful if tracking/keypoint recall drops
  under the default end-to-end head.
- This repo uses the Ultralytics Python API, so raw tensor output changes should
  be hidden by `Results`. Still smoke-test that `results[0].boxes.id`,
  `boxes.xyxy`, and `keypoints.data` are populated before full validation.
- Do not confuse detector NMS with the repo's punch-event NMS. Even if YOLO26
  detector inference is NMS-free, `rascar_boxing.pose_heuristic.select_candidates`
  still needs temporal grouped NMS over punch candidates.
- Because `tools/run_pose_baseline.py` currently passes only `device`, `classes`,
  `conf`, and `verbose` into `model.track(...)`, the first YOLO26 extraction
  should use the official default end-to-end behavior. If the smoke shows low
  recall or unstable IDs, add a future extractor option to pass
  `end2end=False`, then repeat the `n` smoke before testing `s/m/l`.

## GPU1 Commands

Use `--cuda-visible-devices 1` and omit `--device`. Confirm long jobs with:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

Physical GPU 1 UUID from `HYPOTHESES.md`:
`GPU-ff3c1fe8-747e-e4c7-a604-5088eee8c87e`.

Validation split keys:

```text
agn_003,agn_004,agn_010,agn_023,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071,agn_072
```

Smoke one validation video, 300 frames:

```bash
python3 tools/run_pose_batch.py --data-root data/raw --videos-csv data/raw/train/videos.csv --video-keys agn_003 --output-dir data/processed/pose_tracks/smoke_yolo26n_conf035 --model yolo26n-pose.pt --conf 0.35 --cuda-visible-devices 1 --jobs 1 --max-frames 300 --no-tqdm
```

Full validation extraction shape:

```bash
python3 tools/run_pose_batch.py --data-root data/raw --videos-csv data/raw/train/videos.csv --video-keys agn_003,agn_004,agn_010,agn_023,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071,agn_072 --output-dir data/processed/pose_tracks/val_yolo26s_conf035 --model yolo26s-pose.pt --conf 0.35 --cuda-visible-devices 1 --jobs 1 --no-tqdm
```

Use the same command with `val_yolo26n_conf035`, `val_yolo26m_conf035`, or
`val_yolo26l_conf035` and the matching model name. Keep `jobs=1` for `m/l`;
raise jobs only after checking GPU 1 memory.

Test extraction only after validation gates pass:

```bash
python3 tools/run_pose_batch.py --data-root data/raw --videos-csv data/raw/test/videos.csv --output-dir data/processed/pose_tracks/test_yolo26s_conf035 --model yolo26s-pose.pt --conf 0.35 --cuda-visible-devices 1 --jobs 1 --no-tqdm
```

## Validation Order

1. Complete-track check: every validation JSONL must have exactly the manifest
   `frame_count`; reject partial outputs before scoring.
2. Raw grouped-NMS grid, starting from the known best surface:

```bash
python3 tools/evaluate_pose_selection_variants.py --tracks-dir data/processed/pose_tracks/val_yolo26s_conf035 --group-modes fighter_hand --thresholds 0.6,0.8,0.9,1.0,1.15,1.3,1.5 --same-nms-frames 8,10,12 --cross-nms-frames 2,4 --count-modes threshold,root_count,root_round_count --count-multipliers 0.8,0.9,1.0 --top-k 30
```

3. Temporal-context grid only if raw grouped NMS is close to or above
   `0.340888`. Include the `yolo11m` winning family and `yolo11s` control:

```bash
python3 tools/evaluate_pose_temporal_context.py --tracks-dir data/processed/pose_tracks/val_yolo26s_conf035 --features same_sum,dominance --windows 4,6,8 --alphas -0.2,0.0,0.1,0.2 --thresholds 0.8,0.9,1.0,1.15,1.3 --nms-frames 8,10,12 --nms-group-modes fighter_hand --cross-nms-frames 2,4 --top-k 30
```

4. Repeat steps 1-3 for `m` only after `s` looks sane. Repeat for `l` only
   after `m` passes the gates or shows a clear recall-specific failure that `l`
   could plausibly fix.
5. Generate a test CSV only for the best validation candidate, not for every
   threshold sweep:

```bash
python3 tools/make_pose_heuristic_submission.py --tracks-dir data/processed/pose_tracks/test_yolo26s_conf035 --output submissions/yolo26s_fighterhand_thr115_same10_cross4_rootcount1_OFFLINE_CANDIDATE.csv --min-score 1.15 --nms-group-mode fighter_hand --nms-frames 10 --cross-nms-frames 4 --count-mode root_count --count-multiplier 1.0
python3 tools/validate_data.py --submission submissions/yolo26s_fighterhand_thr115_same10_cross4_rootcount1_OFFLINE_CANDIDATE.csv
```

Adjust the filename and parameters to the actual winning grid point.

## Compatibility Risks

- First YOLO26 load may download weights; record exact package version and file
  path before comparing scores.
- End-to-end default may change box/keypoint density and tracker IDs relative
  to YOLO11. The role manager depends on stable person tracks, not just
  per-frame detections.
- If `model.track(..., end2end=False)` is needed, this repo needs a small
  extractor change before running full validation. That is outside this note's
  write scope.
- Higher COCO pose mAP may not transfer to boxing punch timing. `yolo11m` is
  the warning case: better offline detector features did not improve public.
- Prediction-count inflation is dangerous. Dense local validation candidates
  can improve time score while increasing public FP risk.
- `yolo26l` may be too slow or memory-heavy for multi-job extraction on GPU 1.
  Keep `jobs=1` and avoid concurrent GPU work.

## Gates Before Kaggle

- Beat the current `yolo11s` validation baseline `0.340888` by at least
  `+0.010` macro, or present a manual-review case with clearly better timing
  and no count inflation.
- FP penalty must not regress by more than `0.01` versus the matching `yolo11s`
  baseline.
- Require at least 10 of 13 validation-video wins versus the current best
  `yolo11s` candidate, not just versus `yolo11n`.
- The best candidate must be materially different from the failed-public
  `yolo11m` context/count tweak. A bigger detector plus the same overfit count
  behavior is not enough.
- Test CSV must pass `tools/validate_data.py --submission ...` and keep 1594
  fixed rows with unused rows as `clear=false`.
- Clear-true count should be plausible relative to the successful `yolo11s`
  test candidate (`730`) and the known public-risk `yolo11m` result. Large
  count changes need manual review.
- Submit at most one YOLO26 candidate per materially new weight/config. Do not
  spend Kaggle submissions on threshold/NMS sweeps.
