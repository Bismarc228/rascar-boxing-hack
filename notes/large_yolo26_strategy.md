# Large YOLO26 Validation Strategy

Context: `yolo26s` full validation topped out around `0.305323`, well below the
current public-transfer anchor `yolo11s` validation score `0.340888`. The main
process is expected to be extracting `val_yolo26m_conf035`; do not score or
compare that cache until all 13 validation JSONL files are complete. This note
is strategy only: no GPU work, no test extraction, and no Kaggle submission from
this planning pass.

## Operating Rules

- Use physical GPU 1 only for any future extraction:
  `--cuda-visible-devices 1`, omit `--device`, and verify with
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
  Never touch GPU 0.
- Do not run another extraction while `val_yolo26m_conf035` is still being
  written. Partial JSONL files can pass superficial directory checks and poison
  validation comparisons.
- For `yolo26m/l/x`, default to `--jobs 1 --no-tqdm`. Raise jobs only after a
  finished run proves memory headroom and GPU 1 routing; do not use multi-job
  `x` unless there is a measured reason.
- Keep one variable active per round. Do not combine a new model size, detector
  confidence, `end2end=False`, and tracker changes in the same validation run.
- Treat validation score as necessary but insufficient. The known `yolo11m`
  failure (`0.363325` local, worse public) means larger YOLO26 models need a
  transfer-risk story: stable tracks, plausible counts, low FP penalty, and
  wins across videos/roots.
- Limit evaluator output with `--top-k 20` or `--top-k 40`; redirect full logs
  to a per-run log file and inspect only the first readiness lines plus the top
  score table. The goal is to compare summaries, not stream every grid row.

## First Gate: Finish And Audit `yolo26m`

Do not start `l` or `x` until `val_yolo26m_conf035` passes these checks:

1. Complete-cache gate: all 13 validation videos are present, non-empty, and
   have line counts equal to `data/raw/train/videos.csv` `frame_count`.
2. Track-quality audit: compare both-role coverage, red/blue presence, and
   role `track_id` switch rate against the known pattern:

```text
model    both_roles  red_present  blue_present  red_sw/1k  blue_sw/1k
yolo11s  0.8430      0.9326       0.9057        11.20      11.30
yolo11m  0.8693      0.9405       0.9235         8.65       7.90
yolo26s  0.7948      0.9062       0.8796        20.18      16.53
```

3. Score-scale audit: inspect raw candidate count, no-threshold grouped-NMS
   count, score percentiles, and counts above `0.65/0.85/1.05/1.15`. If
   `yolo26m` still looks like `yolo26s` with lower density and unstable tracks,
   bigger weights are not automatically justified.
4. Narrow validation grid first:

```bash
python3 tools/evaluate_pose_selection_variants.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26m_conf035 \
  --group-modes fighter_hand \
  --thresholds 0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15 \
  --same-nms-frames 8,10,12 \
  --cross-nms-frames 2,4,6 \
  --count-modes threshold,root_count,root_round_count \
  --count-multipliers 0.8,0.9,1.0 \
  --top-k 40
```

5. Run temporal context only if the raw grouped grid is close to the anchor
   score or has a credible timing/count explanation:

```bash
python3 tools/evaluate_pose_temporal_context.py \
  --tracks-dir data/processed/pose_tracks/val_yolo26m_conf035 \
  --features same_sum,dominance \
  --windows 4,6,8 \
  --alphas -0.2,0.0,0.1,0.2 \
  --thresholds 0.65,0.75,0.85,0.95,1.05,1.15 \
  --nms-frames 8,10,12 \
  --nms-group-modes fighter_hand \
  --cross-nms-frames 2,4 \
  --top-k 30
```

Decision after `m`:

- Stop the large-YOLO26 branch if `m` is below `0.315` and has `yolo26s`-like
  track quality. That says the family/default extraction path is the issue, not
  model capacity.
- Consider one diagnostic `conf` or `end2end` run if `m` improves cache quality
  but remains below `0.340888`.
- Promote to `l` if `m` is within about `0.015` macro of `0.340888`, beats
  `yolo26s` clearly, or materially fixes the coverage/switch-rate problem.
- Do not promote because of a threshold-only count increase. If `n_pred` jumps
  and FP penalty worsens, treat it like the failed-public `yolo11m` pattern.

## Model Order

1. `yolo26m_conf035` current run: finish, audit, and score before any new
   branch.
2. `yolo26l_conf035`: run only after `m` passes the promotion rule. Use
   `jobs=1`, default YOLO26 end-to-end behavior, and the same validation split.
   Score with the same narrow grid so the comparison isolates model size.
3. `yolo26x_conf035`: run only if `l` creates a real decision:
   - `l` beats `0.340888` without count/FP inflation but not enough for the
     automatic submit gate, or
   - `m -> l` shows a monotonic improvement in score and track quality, making
     `x` a plausible final validation check.

Skip `x` if `l` merely reproduces `m`, wins only through more predictions, or
keeps `yolo26s`-style unstable tracks. The largest model is allowed, but it
should answer a specific question rather than expand the search grid.

## Detector Confidence Strategy

`conf=0.35` may not be equivalent across YOLO11 and YOLO26. For `yolo26s`, it
produced fewer raw candidates, fewer high-score candidates, and lower
both-role coverage than `yolo11s`. Still, do not sweep confidence across every
model size.

Use this order:

1. Finish default `conf035` for the current model.
2. If the failure is recall-limited, run one lower-confidence validation cache
   on the smallest promising model, usually `yolo26m_conf025`.
3. If `conf025` improves coverage but is noisy, try `conf030`; if it worsens
   switches/FP pressure, stop the confidence branch.
4. Try higher confidence such as `conf045` only if the candidate already beats
   the score anchor but loses on FP penalty or implausible prediction count.
5. For `l` and `x`, inherit an alternate confidence only if the `m` confidence
   run improved both cache quality and validation macro. Otherwise use `0.35`.

Gate for a full alternate-conf run: it must plausibly fix a measured issue
from `conf035`. Do not run `0.20/0.25/0.30/0.35/0.45` for every weight.

## End2End Strategy

The current extractor does not expose an `--end2end` flag, so all existing
YOLO26 validation caches use Ultralytics' default YOLO26 behavior. A future
`end2end=False` experiment needs a small extractor change before full
validation.

Use `end2end=False` only as a diagnostic if default YOLO26 has poor coverage or
high switch rates:

1. Add the extractor option in a separate scoped change.
2. Smoke one or two validation videos with `yolo26m`, same `conf`, same tracker,
   and `jobs=1`.
3. Compare schema, line counts, both-role coverage, switch rates, and raw
   candidate density against the default cache.
4. Full 13-video validation is justified only if the smoke improves track
   quality without obvious candidate explosion.
5. Do not combine `end2end=False` with a confidence sweep. First compare
   default versus non-default head at the same `conf`.

Only port `end2end=False` to `l` or `x` if it beats the default `m` run on both
track quality and validation macro.

## Tracker Settings

`tools/run_pose_baseline.py` currently calls `model.track(..., persist=True,
classes=[0], conf=..., verbose=False)` and does not expose a tracker YAML
switch. Keep `persist=True`; the pose heuristic relies on short wrist histories
and role continuity.

Tracker tuning is lower priority than model size and detector confidence:

- Add a tracker option only if `m/l` show good per-frame pose density but high
  role switch rates.
- Smoke-test tracker changes on the same small validation subset before any
  full run.
- Promote a tracker variant to full validation only if switch rates improve by
  roughly `20%+` without reducing both-role coverage or raw high-score
  candidate density.
- Do not tune tracker YAML and detector confidence at the same time.

## Output Discipline

For future long jobs, prefer log files over live console spam:

```bash
python3 tools/run_pose_batch.py ... --cuda-visible-devices 1 --jobs 1 --no-tqdm \
  > logs/val_yolo26l_conf035.log 2>&1
```

For validation grids, keep only the useful summaries:

```bash
python3 tools/evaluate_pose_selection_variants.py ... --top-k 40 \
  > logs/eval_val_yolo26l_conf035_grid.log 2>&1
tail -n 60 logs/eval_val_yolo26l_conf035_grid.log
```

The score table should record at least: macro score, time score, FP penalty,
wins, `n_pred`, group mode, threshold, same/cross NMS, count mode, multiplier,
and frame offset. If two settings differ only by a tiny threshold/NMS tweak,
keep the simpler and more conservative one.

## Gates Before Test Extraction

Test extraction is justified only after a single validation candidate passes a
hard gate or a documented manual-review gate.

Hard gate:

- Complete validation cache for the exact model/config.
- Macro score at least `0.350888` (`yolo11s 0.340888 + 0.010`).
- At least 10 of 13 validation-video wins versus the current `yolo11s` anchor.
- FP penalty regression no worse than `0.01`.
- `n_pred` and clear-true count are plausible, not a large count-inflation win.
- Track-quality metrics do not look worse than `yolo11s` in a way that would
  threaten public transfer.

Manual-review gate:

- Macro is above `0.340888` but below the hard gate.
- Timing improves clearly, FP penalty is flat or better, and the win is not
  produced by lower thresholds alone.
- Results are stable across validation roots/videos and do not resemble the
  failed-public `yolo11m` context/count pattern.

Do not extract test for:

- `yolo26m/l/x` runs below the `yolo11s` validation anchor.
- Pure confidence, tracker, or `end2end` diagnostics that only explain failure.
- Multiple threshold/NMS variants from the same cache.

When test extraction is justified, generate exactly one test cache for the best
validation model/config and one CSV from the selected parameters. Validate the
CSV locally with `tools/validate_data.py --submission ...`. Kaggle submission is
a separate decision, not implied by this note.
