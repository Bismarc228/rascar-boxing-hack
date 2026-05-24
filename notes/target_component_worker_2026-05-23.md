# Target Component Worker - 2026-05-23

Goal: verify revived hypothesis #6, pose-geometry target-only correction, on
the current strict OOF anchor. No Kaggle upload was made.

## Scope

Writable files used by this worker:

- `tools/evaluate_target_component_guard.py`
- `notes/target_component_worker_2026-05-23.md`
- `data/processed/diagnostics/target_component_guard_strict_20260523_*`
- `data/processed/vit_features/target_component_*_20260523.csv`

Inputs:

- strict OOF anchor:
  `data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv`
- pose tracks:
  `data/processed/pose_tracks/val_yolo11s_conf035/`
  `data/processed/pose_tracks/val_yolo26l_conf035/`
  `data/processed/pose_tracks/val_yolo26x_conf035/`
- prior root-out target evidence:
  `notes/fixed_row_target_attribute_smoke_2026-05-22.md`

## Commands

Raw unguarded target-only control:

```bash
python3 tools/evaluate_fixed_row_attribute_model.py \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 \
  --write-variant target \
  --write-oof-rows data/processed/vit_features/target_component_raw_target_strict_oof_20260523.csv
```

Guarded target component evaluator:

```bash
python3 -m py_compile tools/evaluate_target_component_guard.py
python3 tools/evaluate_target_component_guard.py \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --diagnostics-prefix data/processed/diagnostics/target_component_guard_strict_20260523 \
  --write-selected-rows data/processed/vit_features/target_component_guarded_target_strict_oof_20260523.csv
python3 tools/score_predictions.py \
  --predictions data/processed/vit_features/target_component_guarded_target_strict_oof_20260523.csv \
  --video-keys-from-predictions
git diff --check
```

## Strict OOF Results

Baseline:

```text
macro=0.415261
target=0.478143
hand=0.521370
rows=1188
```

Raw unguarded target-only control:

```text
macro=0.414847
delta=-0.000414
target=0.471237
target_delta=-0.006906
changed=124
```

Best guarded target-only result:

```text
macro=0.415489
delta=+0.000227
target=0.481934
target_delta=+0.003790
hand=0.521370
hand_delta=+0.000000
wins=4
changed=11
guard=effectiveness:miss, transition:both, min_margin:0.25, min_stable_votes:2
public_changed=0
```

Fixed-row assertions:

```text
changed_columns: target=11, all other submission columns=0
hand_unchanged_assertion=True
fixed_target_only_assertion=True
```

Negative hand control:

```text
macro=0.414654
delta=-0.000607
hand=0.511249
hand_delta=-0.010121
hand_changed=39
```

## Target Confusion

Scorable matched target confusion:

```text
variant   actual  pred  n
baseline  body    body   69
baseline  body    head  164
baseline  head    body   49
baseline  head    head  762
selected  body    body   70
selected  body    head  163
selected  head    body   44
selected  head    head  767
```

The selected guard mostly removes `head -> body` mistakes and adds one
`body -> body` recovery, but the total component lift is small.

## Root And Video Deltas

Tournament-root deltas are non-negative:

```text
dataset_type   data_root       delta_macro  delta_target  wins  losses
tournament_1   Tournament 1    +0.000315    +0.005249     1     0
tournament_2   Tournament 2    +0.000201    +0.003345     2     0
training       training root   +0.000202    +0.003372     1     1
```

Only one video regressed: `agn_004`, with `delta_macro=-0.000504` and
`delta_target=-0.008403`. Tournament videos were either wins or ties.

## Public-Sensitive Control

The strict validation anchor contains no `agn_037`, `agn_038`, or `agn_039`
rows, so `public_changed=0`. The evaluator still includes an `exclude_public`
policy in the guard grid for fixed-row controls on row sets that contain those
videos.

## Decision

Fail against the revived-hypothesis pass gate on the current strict OOF anchor.

Pass requirements were:

- `score_target +0.020` or `macro +0.0012`
- hand unchanged
- non-negative tournament-root deltas

Observed:

- `score_target +0.003790`
- `macro +0.000227`
- hand unchanged
- tournament-root deltas non-negative

Interpretation: the old root-out target-only signal was real on the previous
root-out source, but it does not transfer with enough magnitude to the current
strict anchor. A safe target-only guard exists, yet it is too weak to promote.
