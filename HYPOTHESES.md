# Hypotheses Backlog

Research backlog for the Kaggle `boxing-action-recognition-challenge`. Evidence
comes from `EXPERIMENTS.md`, `OVERVIEW.md`, and `DATA_DESCRIPTION.md`.

## Current State

- Best public score: `0.13275` from
  `submissions/yolo11s_fighterhand_thr115_same10_cross4_rootcount1_OFFLINE_CANDIDATE.csv`.
- Corrected offline validation uses all 13 fight-level validation videos in
  `data/processed/pose_tracks/val_yolo11n_conf035/`.
- `yolo11n` simple offline threshold/NMS baseline:
  `0.23802` at `threshold=0.8`, `nms=6`, `n=1119`.
- Best `yolo11n` submitted-public analog on the same split:
  `0.20130` at `threshold=1.0`, `nms=12`.
- Best checked grouped NMS:
  `0.24246`, fighter-grouped NMS, `threshold=0.8`, same-group NMS `8`,
  cross-group NMS `2`, `n=1151`.
- Best checked grouped count variant:
  `0.242684`, root-count multiplier `0.9`.
- Best checked temporal-context variant:
  `0.245871`, same `(fighter, hand)` dominance in a `+/-4` frame window,
  `alpha=0.2`, `threshold=1.05`, same-fighter NMS `7`, cross-fighter NMS `2`,
  `n=1170`.
- Best checked `yolo11s-pose` offline variant:
  `0.340888`, fighter+hand grouped NMS, `threshold=1.15`, same-group NMS `10`,
  cross-group NMS `4`, `root_count=1.0`, `n=1202`.
- Best checked `yolo11m-pose` offline variant:
  `0.363325`, same-score temporal context in a `+/-8` frame window,
  `alpha=-0.2`, fighter+hand grouped NMS, `threshold=0.9`, same-group NMS `10`,
  cross-group NMS `4`, `n=1198`, FP penalty `0.065930`, 11/13 validation wins.
  Public score for the matching candidate was only `0.12958`, so it does not
  replace the `yolo11s` public baseline.
- Generated but not submitted candidates include the grouped-NMS, grouped-count,
  and temporal-context CSVs listed in `EXPERIMENTS.md`.
- The task rewards timing most heavily. The metric weights time at `50%`,
  fighter at `20%`, punch type at `10%`, effectiveness at `8%`, and hand/target
  at `6%` each, with a separate FP penalty. Rows with `clear=false` do not match
  and are not counted as FP.
- Test has 9 videos and the submission template has 1594 fixed rows. Use
  `clear=false` for unused rows and always pass local schema validation before
  considering a submit.

## Hypotheses Already Killed

- Sample submission frames/counts are not labels.
- Pure temporal priors are too weak; the temporal-prior and sample baselines
  scored `0.00000` publicly.
- Global fighter swap, all-red/all-blue, and confidence-threshold role fixes do
  not beat the current role assignment.
- Global frame offsets do not fix timing. Offset `0` is best around grouped NMS;
  `+1`, `-1`, and larger shifts degrade.
- Supervised rankers using the current pose features underperform the simple
  heuristic on mean LOOV.
- Supervised attribute models only produced a small gain in one timing setup
  (`0.21299 -> 0.21590`), not enough to submit.
- Motion-only candidates are noisy. Crop-motion reranking is more plausible, but
  the first grid did not beat the full-split threshold/NMS candidate.
- Audio-only onset detection is not a detector (`0.02927` macro with oracle
  counts), and hard-shifting pose candidates to audio peaks hurts.
- Audio reranking has not beaten the full-validation baseline; the best positive
  audio boost was still slightly worse than baseline (`0.23789` vs `0.23802`).
- Fighter/color and attribute priors are not useful standalone improvements.
- `yolo11s-pose` transferred to public and is now the strongest baseline:
  `0.13275` public. The next detector/model experiments should compare against
  `yolo11s`, not `yolo11n`.
- `yolo11m-pose` beats `yolo11s` offline by more than the submit gate, but the
  first public `yolo11m` context candidate scored `0.12958` versus `0.13275`.
  Treat this as overfit until a stronger validation explanation exists.
- `yolo26s-pose` is not useful as a replacement detector: validation is only
  `0.305323`, and role coverage/track-switch metrics are worse than the YOLO11
  caches.
- `yolo26m-pose` is a possible secondary timing witness but not a submit path
  yet. Its best context candidate reaches `0.352804` with good FP penalty, but
  the gain is concentrated in the old `бокс` root while both tournament roots
  fail to improve versus the public-proven `yolo11s` anchor.
- For larger pose models, prefer asymmetric agreement/snapping/fusion against
  `yolo11s` over replacing the detector wholesale. A large-model-only candidate
  must pass tournament-root audit, not just full-split macro.

## Promising Next Hypotheses

- Expand temporal-context scoring around pose candidates, not single-frame
  scoring. The current best signal is local same `(fighter, hand)` dominance.
- Train an event selector only if it uses stronger temporal features and
  fight-group validation. Keep the simple heuristic as fallback.
- Improve per-video count control. Dense NMS improves offline, but test/public
  FP risk may differ by fight, so count calibration needs stress tests.
- Use audio only as a weak learned feature such as local onset max or nearest
  onset distance inside a pose-dominant selector. Do not hard-shift frames.
- Revisit crop-motion only as a tie-breaker/reranker for a wide pose pool.
- Delay attribute work until timing/selection improves; attributes have lower
  metric weight and have not yet moved enough offline.

## Compute-Aware Experiment Queue

1. Public-overfit audit for `yolo11m`.
   - Compare the submitted context candidate against raw `yolo11m` and
     `yolo11s` on validation by video, prediction count, and sample-capacity
     clipping.
   - Do not submit another `yolo11m` threshold/context tweak until that audit
     identifies a materially different fix.
2. No-GPU offline grid: rerun and narrow temporal-context sweeps on cached
   `yolo11s` and later `yolo11m` validation tracks.
   - Start near `window=4`, `alpha=0.2`, `dominance`,
     `threshold=1.0..1.15`, same-fighter NMS `6..8`, cross-NMS `2..4`.
   - Command shape:
     `python3 tools/evaluate_pose_temporal_context.py --tracks-dir data/processed/pose_tracks/val_yolo11n_conf035 --windows 3,4,5,6 --alphas 0.1,0.15,0.2,0.25,0.35 --features dominance,same_sum --thresholds 0.95,1.0,1.05,1.1,1.15 --nms-frames 6,7,8 --nms-group-modes fighter --cross-nms-frames 2,3,4`.
3. No-GPU count stress test on cached validation tracks.
   - Compare threshold-only against root/root-round count modes around
     multipliers `0.8..1.0`.
   - Command shape:
     `python3 tools/evaluate_pose_threshold_grid.py --tracks-dir data/processed/pose_tracks/val_yolo11n_conf035 --thresholds 0.7,0.8,0.9,1.0,1.05,1.15 --nms-frames 4,6,7,8,10 --count-modes threshold,root_count,root_round_count,root_rate,root_round_rate --count-multipliers 0.8,0.9,1.0`.
4. No-GPU robustness audit.
   - Compare macro, time score, FP penalty, prediction count, and video wins
     against both `0.23802` and `0.245871`. Reject variants that win by count
     inflation only.
5. GPU pass only if tracks are missing or a new detector/config is justified.
   - Regenerate pose tracks with GPU 1 only, verify with `nvidia-smi` UUID, then
     return to cached offline evaluation.
6. Candidate generation after a gate passes.
   - Use `tools/make_pose_heuristic_submission.py` with the winning parameters.
   - Validate locally; do not submit from this backlog step.
7. Large-model witness tests.
   - Finish `yolo26l`/future `yolo26x` validation caches on GPU 1, then run the
     same complete-cache, track-quality, score-scale, and tournament-root gate.
   - If a large model is only strong on old `бокс`, use it only for fixed-count
     timing snap or agreement boost around `yolo11s` events.

## Submit Gates

- Respect the 30 submissions/day limit. Do not submit small threshold sweeps.
- A candidate must pass local validation against `sample_submission.csv`.
- A candidate must be materially different from already submitted variants.
- Automatic submit gate: beat the current best offline `yolo11s` score
  `0.340888` by at least `0.010` macro, with no FP-penalty regression larger
  than `0.01` and at least 10 of 13 validation-video wins versus
  `thr=0.8,nms=6`.
- Manual-review gate: a smaller gain may be worth discussing only if it improves
  timing score clearly, keeps prediction counts plausible, and is not another
  threshold/NMS-only tweak.
- Never submit audio-only, pure temporal-prior, global offset, or attribute-only
  variants unless they are part of a larger pose-dominant candidate that passes
  the gates above.

## Use GPU 1 Only

- For Ultralytics pose extraction in this repo, use `CUDA_VISIBLE_DEVICES=1`
  and **omit** `--device`.
- Do not combine `CUDA_VISIBLE_DEVICES=1` with `--device 0`; this was observed
  to route Ultralytics to physical GPU 0 on this machine.
- Do not rely on `--device 1` alone for Ultralytics here; this version can
  rewrite visibility internally and fail or select the wrong device.
- Confirm any long GPU job with:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
  Physical GPU 1 UUID is `GPU-ff3c1fe8-747e-e4c7-a604-5088eee8c87e`.
- For the batch pose extractor, keep the built-in GPU-1 visibility default and
  do not pass `--device`:
  `python3 tools/run_pose_batch.py --cuda-visible-devices 1 --jobs 3 ...`.
- Example validation-track extraction:
  `python3 tools/run_pose_batch.py --data-root data/raw --videos-csv data/raw/train/videos.csv --output-dir data/processed/pose_tracks/val_yolo11n_conf035 --model models/yolo11n-pose.pt --conf 0.35 --cuda-visible-devices 1 --jobs 3 --no-tqdm`.
- Most backlog steps use cached tracks and do not need GPU.
