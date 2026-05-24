# Calibrated Fighter Flip Worker - 2026-05-23

Scope: revived hypothesis #4, fixed selected rows only. No Kaggle upload.
Only `fighter` is changed in candidate rows.

Anchor:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Baseline on the 13 strict OOF videos:

```text
macro=0.415261
score_fighter=0.537953
time=0.530111
fp_penalty=0.061715
rows=1188
```

## Tool

Added:

```text
tools/evaluate_calibrated_fighter_flip_gate.py
```

The evaluator combines:

- existing dual-attacker OOF row signal:
  `dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv`;
- local opposite-fighter pose rival evidence from cached pose tracks;
- cached per-video appearance calibration summary when available:
  `train_fighter_appearance_video_summary.csv`.

It writes strict gates with at most 30 fighter flips and reports macro,
`score_fighter`, time, FP penalty, flips by video/root, root deltas, and a
flip-level audit.

## Commands

Primary cached train-yolo11s pose sweep:

```bash
.venv/bin/python tools/evaluate_calibrated_fighter_flip_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --appearance-summary data/processed/appearance_audits/dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv \
  --output-prefix data/processed/diagnostics/calibrated_fighter_flip_train_yolo11s_20260523
```

Cached val-yolo26l pose sweep:

```bash
.venv/bin/python tools/evaluate_calibrated_fighter_flip_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --appearance-summary data/processed/appearance_audits/dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv \
  --output-prefix data/processed/diagnostics/calibrated_fighter_flip_val_yolo26l_20260523
```

Narrow broader-rival check:

```bash
.venv/bin/python tools/evaluate_calibrated_fighter_flip_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --appearance-summary data/processed/appearance_audits/dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv \
  --output-prefix data/processed/diagnostics/calibrated_fighter_flip_broad_val_yolo26l_20260523 \
  --sources dual_or_rival,rival \
  --windows 0,2,4 \
  --match-modes same_hand_target,same_hand \
  --ratios 1.6,2.0 \
  --min-rival-scores 0.8,1.0 \
  --top-ns 5,10,15,20,25,30 \
  --rank-metrics pose_diff,appearance_pose_diff \
  --appearance-gates calibration_ok,color_early_acc_ge_0.85
```

## Results

Best strict rows:

```text
run                         macro     delta     fighter   delta_fighter  flips  time      fp        root risk
train_yolo11s default        0.415953  +0.000692 0.541502 +0.003549      25     0.530111  0.061715  none
val_yolo26l default          0.416120  +0.000859 0.542000 +0.004047      20     0.530111  0.061715  none
broad val_yolo26l high-rival 0.415717  +0.000456 0.539984 +0.002031      30     0.530111  0.061715  tournament_2 -0.000170
```

No swept row passed the configured gate:

```text
train_yolo11s default:        130 rows, pass_gate=0
val_yolo26l default:          119 rows, pass_gate=0
broad val_yolo26l high-rival: 236 rows, pass_gate=0
```

Best overall strict candidate:

```text
source=dual
tracks=val_yolo26l_conf035
window=4
match_mode=any
rank_metric=pose_diff
appearance_gate=all
top_n=20
macro=0.416120  delta=+0.000859
score_fighter=0.542000  delta=+0.004047
time=0.530111  delta=+0.000000
fp_penalty=0.061715  delta=+0.000000
wins=6
```

Flips by video:

```text
agn_023:4
agn_025:1
agn_056:1
agn_058:1
agn_069:3
agn_070:3
agn_071:3
agn_072:4
```

Flips by tournament root:

```text
tournament_1|Турнир Бокс:5
tournament_2|Турнир Бокс 2:15
```

Root deltas for the best strict candidate:

```text
tournament_1|Турнир Бокс    macro +0.001549  fighter +0.006665  time +0.000000  fp +0.000000
tournament_2|Турнир Бокс 2  macro +0.000932  fighter +0.004659  time +0.000000  fp +0.000000
training|бокс               macro +0.000000  fighter +0.000000  time +0.000000  fp +0.000000
```

Flip audit for the best strict candidate:

```text
flips=20
dual_flips=20
transitions: blue->red=14, red->blue=6
OOF matched-fighter outcomes: fix=12, break=4, unmatched=4
```

The best candidate is real but too small: the local pose ranking removes some
bad dual flips, but it cannot turn the dual-attacker signal into the required
component lift. The broader high-confidence rival check is weaker and creates
a negative `tournament_2|Турнир Бокс 2` root delta.

## Artifacts

```text
data/processed/diagnostics/calibrated_fighter_flip_dual_by_video_precheck_20260523.csv

data/processed/diagnostics/calibrated_fighter_flip_train_yolo11s_20260523_summary.csv
data/processed/diagnostics/calibrated_fighter_flip_train_yolo11s_20260523_best_rows.csv
data/processed/diagnostics/calibrated_fighter_flip_train_yolo11s_20260523_best_flips.csv
data/processed/diagnostics/calibrated_fighter_flip_train_yolo11s_20260523_best_root_deltas.csv

data/processed/diagnostics/calibrated_fighter_flip_val_yolo26l_20260523_summary.csv
data/processed/diagnostics/calibrated_fighter_flip_val_yolo26l_20260523_best_rows.csv
data/processed/diagnostics/calibrated_fighter_flip_val_yolo26l_20260523_best_flips.csv
data/processed/diagnostics/calibrated_fighter_flip_val_yolo26l_20260523_best_root_deltas.csv

data/processed/diagnostics/calibrated_fighter_flip_broad_val_yolo26l_20260523_summary.csv
data/processed/diagnostics/calibrated_fighter_flip_broad_val_yolo26l_20260523_best_rows.csv
data/processed/diagnostics/calibrated_fighter_flip_broad_val_yolo26l_20260523_best_flips.csv
data/processed/diagnostics/calibrated_fighter_flip_broad_val_yolo26l_20260523_best_root_deltas.csv
```

## Decision

Fail revived hypothesis #4 under the requested pass rule.

Pass rule was `score_fighter +0.010` or `macro +0.002`, at most 30 flips,
time/FP unchanged, and no negative tournament-root macro. The best strict
candidate satisfies the flip, time/FP, and root-risk constraints, but reaches
only `score_fighter +0.004047` and `macro +0.000859`.

Keep the existing dual-attacker fighter branch as a small independent
micro-signal/risk reference. Do not promote this calibrated fighter-flip gate
as a standalone candidate.

## Dual Proposal Coverage Follow-Up

The matched-fighter oracle on the current strict anchor confirms there is real
fighter headroom:

```text
baseline_macro=0.415261
baseline_fighter=0.537953

oracle_matched_fighter_macro=0.429137
oracle_matched_fighter_delta=+0.013876
oracle_matched_fighter_score_fighter=0.607932
oracle_matched_fighter_delta_fighter=+0.069979
matched_scorable=1044
fighter_wrong=120
```

However the DINO dual-fighter proposal source does not cover enough of that
headroom:

```text
dual_all_macro_delta=+0.000571
dual_all_fighter_delta=+0.002849
dual_changed=32
dual_fix=16
dual_break=11
dual_unmatched=5

oracle_dual_fix_only_selected=16
oracle_dual_fix_only_macro_delta=+0.001575
oracle_dual_fix_only_fighter_delta=+0.007877
```

Artifact:

```text
data/processed/diagnostics/calibrated_fighter_flip_dual_proposal_coverage_20260523.csv
```

Interpretation: even a perfect gate over the current dual-fighter proposals
cannot meet the `fighter_delta=+0.010` pass rule. The blocker is not mainly the
calibration gate; it is proposal recall. Future fighter work needs a new
proposal source that covers more of the `120` wrong-fighter matched rows,
likely from candidate-pool/track identity, not another threshold sweep over the
same dual rows.

## Rival Proposal Coverage Follow-Up

Added a proposal-source audit:

```text
tools/audit_fighter_proposal_coverage.py
```

Output:

```text
data/processed/diagnostics/fighter_proposal_coverage_rival_dual_20260523.csv
```

Best broad pose-rival proposal source:

```text
source=train_yolo11s_conf035
window=8
match_mode=same_hand
ratio=1.6
min_rival_score=0
proposed=174
fix=41
break=130
time_only=3
oracle_fighter_delta=+0.012115
oracle_macro_delta=+0.002543
all_fighter_delta=-0.031426
all_macro_delta=-0.006392
all_time_delta=-0.000238
all_fp_penalty_delta=+0.000000
```

Then ran a focused pure-rival ranking check with the existing calibrated gate:

```text
data/processed/diagnostics/calibrated_fighter_flip_purerival_train_yolo11s_20260523_summary.csv
```

Best simple ranked row:

```text
top_n=15
rank_metric=pose_diff
appearance_gate=all
fighter_delta=+0.000571
macro_delta=+0.000154
time_delta=+0.000000
fp_penalty_delta=+0.000000
negative_tournament_root=tournament_2|Турнир Бокс 2:-0.000243
```

Interpretation update: the next fighter blocker is not merely proposal recall.
Broad pose-rival proposals have enough fix-only headroom to pass the fighter
component gate, but the fix/break precision is very poor and simple pose
ranking cannot separate them. The next fighter step should be a supervised
fix-vs-break gate over the high-recall rival proposal table.

First supervised fix-vs-break smoke:

```text
tools/evaluate_fighter_rival_proposal_gate.py
data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_20260523_summary.csv
data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_20260523_proposals.csv
```

Result: killed for current pose/metadata/local-count features. The proposal
table has `41` fixes, `130` breaks, and `3` time-only rows, but the best OOF
selected row is a break (`fighter_delta=-0.000455`,
`macro_delta=-0.000091`). HGB probability is actually higher on breaks on
average (`break_mean=0.228863`, `fix_mean=0.194215`), and logreg separation is
too weak. Future fighter gating needs a stronger visual/tracklet/temporal
identity witness.

Adding the existing per-video appearance calibration summary and DINO dual
sidecar as features is also negative: best selected row has
`fighter_delta=-0.000414` and `macro_delta=-0.000083`. Video-level calibration
metadata is not enough; the missing evidence needs to be row-level.
