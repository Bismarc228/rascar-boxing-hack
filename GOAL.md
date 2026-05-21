# Boxing Action Recognition Challenge - Goal

## Objective

Build a solution for Kaggle competition `boxing-action-recognition-challenge`.

For each of 9 test boxing-round videos, predict a fixed-size event table where each row is one punch event. The core task is to detect the punch frame and identify the attacking fighter (`red` or `blue`). Extra punch attributes also affect the score:

- `punch_type`: `jab`, `cross`, `hook`, `uppercut`
- `hand`: `left`, `right`
- `target`: `head`, `body`
- `effectiveness`: `landed`, `blocked`, `miss`
- `clear`: `true`, `false`

Frames are 0-based inside each mp4/MOV file, matching `train/punches.csv`.

## Local Data State

Kaggle archive is unpacked into `data/raw/`.

Available files:

- `data/raw/train/videos.csv`: 63 train clips.
- `data/raw/train/punches.csv`: 9030 labeled punch events.
- `data/raw/test/videos.csv`: 9 test clips.
- `data/raw/sample_submission.csv`: 1594 required submission rows.
- `data/raw/README.txt`: video download links.
- `data/raw/RASCAR_baseline_boxing.ipynb`: Colab baseline for YOLO-pose tracking/role assignment.

Videos are not included in the Kaggle archive. They must be downloaded separately from the three Yandex Disk links in `README.txt` and placed so paths from `video_path` resolve under the dataset/video root:

- `бокс`
- `Турнир Бокс`
- `Турнир Бокс 2`

## Submission Contract

Kaggle expects exactly the same row count and `id` values as `sample_submission.csv`: 1594 rows, `id = 1..1594`. Row order does not affect evaluation.

Required columns:

```csv
id,video_id,agn_index,video_key,frame,fighter,punch_type,hand,target,effectiveness,clear
```

Valid values:

- `fighter`: `red`, `blue`
- `punch_type`: `jab`, `cross`, `hook`, `uppercut`
- `hand`: `left`, `right`
- `target`: `head`, `body`
- `effectiveness`: `landed`, `blocked`, `miss`
- `clear`: `true`, `false`

Important behavior: rows with `clear=false` are not matched and are not counted as false positives. This is useful when keeping the fixed row count while submitting fewer confident punch detections.

## Test Clips

Test videos are three complete fights, all rounds included:

- `АГН-37`, `АГН-38`, `АГН-39`: `Турнир Бокс / Бой9. Вороной-Болотов`
- `АГН-47`, `АГН-48`, `АГН-49`: `Турнир Бокс 2 / Бой 2. Чемоханов и Иванов`
- `АГН-62`, `АГН-63`, `АГН-64`: `Турнир Бокс 2 / Бой 8. Рычков и Львов`

Use `video_key` (`agn_037`, etc.) for ASCII-safe logs/cache paths, and preserve `video_id`, `agn_index`, and `video_key` in submissions.

## Metric

Leaderboard score is macro-average over test videos with `clear=true` ground truth.

Per-video score:

```text
final_score = 0.50 * score_time
            + 0.20 * score_fighter
            + 0.10 * score_punch_type
            + 0.08 * score_effectiveness
            + 0.06 * score_hand
            + 0.06 * score_target
            - fp_penalty
```

The result is clipped to `[0, 1]`.

Matching:

- Hungarian matching by temporal distance.
- Window is +/- 1.0 second using `t = frame / 30`.
- One prediction can match one GT punch.
- If multiple candidates have equal timing, same `fighter` is preferred.
- Only GT with `clear=true` is evaluated.
- Prediction rows with `clear=false` do not match and do not count as FP.

Time error for a matched TP:

```text
time_error = min(1.0, abs(t_gt - t_pred) / 0.5)
```

Interpretation:

- Exact frame/time gives `0.0` error.
- 0.25 s error gives `0.5` error.
- 0.5 s or worse gives max time error.

Attribute scoring:

- `fighter`, `hand`, `target`: simple accuracy-like component over GT punches.
- `punch_type`, `effectiveness`: balanced accuracy with inverse class-frequency weights from GT.
- `effectiveness` is mapped internally as `landed -> hit`, `blocked -> block`, `miss -> miss`.
- FN is max error for all components.
- FP penalty is `N_fp / (N_gt + N_fp)`.

Main metric implication: reliable punch timing and fighter identity matter most. Extra attributes are useful, but over-generating punches is expensive.

## Baseline Direction

Provided notebook `RASCAR_baseline_boxing.ipynb` is not a submission generator. It is a starting point for video understanding:

- Detect people with YOLO-pose.
- Track people across frames.
- Separate `red`, `blue`, referee, and spectators.
- Use HSV color heuristics on torso/shorts/gloves/headgear for fighter role assignment.
- Output annotated video.

Dependencies from the notebook:

- `ultralytics`
- `opencv-python`
- `numpy`
- `scipy`

This baseline should be converted into a reproducible local pipeline that emits CSV, caches intermediate pose/tracking features, and can run on all train/test clips.

## Recommended Work Plan

1. Ingest videos.
   - Download/extract Yandex Disk archives.
   - Make local paths resolve against `data/raw/*/videos.csv`.
   - If Cyrillic paths are fragile, create an ASCII mirror or symlink layout by `video_key`.

2. Build data validation scripts.
   - Check every train/test video exists.
   - Read true FPS from video files with OpenCV/ffprobe.
   - Validate frame counts against CSV.
   - Validate submission schema and allowed categorical values.

3. Port baseline to scripts.
   - Extract YOLO-pose keypoints for all clips.
   - Track two fighters.
   - Assign stable `red`/`blue` roles per video.
   - Cache features under a reproducible path, preferably keyed by `video_key`.

4. Create a local validation split.
   - Split by fight/video, not by event rows.
   - Hold out full fights/rounds to approximate test behavior.
   - Implement the Kaggle metric locally from the description.

5. Build the first submission baseline.
   - Start with timing priors from train/test `sample_submission.csv` row counts only if needed.
   - Better first model: detect high hand-speed/contact candidates from pose tracks.
   - Assign `fighter` from the attacking tracked fighter.
   - Fill secondary attributes with train priors or simple heuristics.
   - Mark unused required rows as `clear=false`.

6. Improve timing and fighter identity.
   - Use temporal windows around candidate frames.
   - Features: wrist velocity/acceleration, distance to opponent head/body, arm extension, torso movement, fighter distance, occlusion/referee indicators.
   - Postprocess with non-max suppression in time to reduce FP.

7. Improve attributes.
   - Train classifiers for `punch_type`, `hand`, `target`, `effectiveness`.
   - Prioritize `punch_type` and `effectiveness` because they are balanced and rare classes matter.
   - Keep conservative `clear=true` decisions to control FP penalty.

## First Success Criteria

- All video files resolve locally.
- A script can generate `submission.csv` with exactly 1594 rows and the required columns.
- Local metric implementation runs on a held-out validation split.
- Baseline submission uses `clear=false` for unused rows and passes Kaggle format validation.
- Iteration loop exists: extract features -> train/evaluate -> generate submit.

## Current Implementation State - 2026-05-20

The initial success criteria are done. The working solution now has local
validation, cached pose features, submission generators, public-score tracking,
and a post-reset candidate queue.

Current public best is `0.16461` from a video-local hybrid: keep the yolo26l
public anchor everywhere except `agn_038`, where sequence-TCN snap4 is better.

```text
submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv
```

Root `submission.csv` is currently copied from this file and locally validated.

Previous public best was `0.13849` from the yolo26l temporal-context/root-rate
branch:

```text
submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
```

The first post-reset yolo26x+yolo11s agreement submit did not transfer and is
killed as a submit branch:

```text
submissions/yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088_OFFLINE_CANDIDATE.csv
offline=0.377949, public=0.10784
```

The strongest learned branch is the pose-sequence TCN with snap-to-local-score
postprocess:

```text
seq_tcn snap4 root_count=0.88: offline=0.389963 first sweep, 0.387039 repeat, rows=744
seq_tcn snap4 root_count=0.92: offline=0.390013, rows=765
seq_tcn snap4 root_count=0.82: offline=0.382612, rows=712
```

Public transfer is video-local: `agn_038` sequence hybrid scores `0.16461`, but
the full `root_count=0.88` sequence submit scores only `0.12712`, and adding
`agn_037` sequence rows to the good hybrid also scores `0.12712`.

The active research direction is no longer the original baseline notebook.
Current priorities are:

- yolo26x agreement and sequence-TCN spotters for punch timing.
- Conservative per-video count and `clear=true` calibration, because public
  punished dense recall.
- Video-local fighter identity with pose-guided tracklet/appearance evidence,
  not whole-video swaps or simple bbox color clustering.
- Audio only as weak local features near pose candidates; audio-only and hard
  audio snapping are killed.
- Submission budget is now capped at at most two more attempts today, only for
  a clear candidate above `0.16461`.
