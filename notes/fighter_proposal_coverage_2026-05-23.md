# Fighter Proposal Coverage - 2026-05-23

Goal: follow up the calibrated fighter-flip failure by checking whether the
blocker is proposal recall or proposal precision. No GPU and no Kaggle upload
were used.

## Tool

Added:

```text
tools/audit_fighter_proposal_coverage.py
```

The auditor scores fixed-row fighter-flip proposal sources before training a
gate. It reports:

- proposal counts and metric-match `fix/break/time_only/unmatched` labels;
- a fix-only oracle proxy for each proposal source;
- the true score if every proposal is applied.

The true metric deltas are the authoritative numbers; the row counts are a
debug view of the proposal source.

## Coverage Sweep

Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/audit_fighter_proposal_coverage.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --windows 0,2,4,8 \
  --match-modes same_hand_target,same_hand,any \
  --ratios 0.8,1.0,1.15,1.3,1.6,2.0 \
  --min-rival-scores 0.0,0.2,0.5,0.8,1.0 \
  --output data/processed/diagnostics/fighter_proposal_coverage_rival_dual_20260523.csv \
  --top-k 30
```

Baseline:

```text
macro=0.415261
fighter=0.537953
time=0.530111
fp_penalty=0.061715
rows=1188
```

Best rival proposal oracle proxy:

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
all_min_video_delta=-0.017043
```

Best true all-proposal row in the sweep is much smaller:

```text
source=val_yolo26x_conf035
window=0
match_mode=same_hand_target
ratio=1.6
min_rival_score=0.8
proposed=11
fix=4
break=7
oracle_fighter_delta=+0.001555
oracle_macro_delta=+0.000311
all_fighter_delta=+0.001106
all_macro_delta=+0.000221
```

The existing DINO dual source remains cleaner but lower-recall:

```text
proposed=32
oracle_fighter_delta=+0.004084
oracle_macro_delta=+0.000817
all_fighter_delta=+0.002849
all_macro_delta=+0.000571
all_time_delta=+0.000000
all_fp_penalty_delta=+0.000000
```

## Simple Ranking Check

Because the high-recall rival source has a pass-level fix-only oracle proxy, I
ran the existing calibrated evaluator on the specific pure-rival setting:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_calibrated_fighter_flip_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --appearance-summary data/processed/appearance_audits/dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv \
  --output-prefix data/processed/diagnostics/calibrated_fighter_flip_purerival_train_yolo11s_20260523 \
  --sources rival \
  --windows 8 \
  --match-modes same_hand \
  --ratios 1.6 \
  --min-rival-scores 0 \
  --top-ns 5,10,15,20,25,30,40,50,60,80,100 \
  --max-flips 100 \
  --rank-metrics pose_diff,pose_ratio,rival_score,appearance_pose_diff \
  --appearance-gates all,calibration_ok
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
flips=15
negative_tournament_root=tournament_2|Турнир Бокс 2:-0.000243
```

## Decision

This changes the fighter diagnosis:

- the broader `train_yolo11s` pose-rival proposal source has enough fix-only
  headroom to pass the fighter component gate;
- naive all-proposal application is very negative, so the source is too noisy;
- simple pose-score/rank/appearance gates do not separate fixes from breaks.

The next fighter step should be a supervised fix-vs-break proposal gate over
the high-recall rival proposal table, not another wider proposal sweep and not
another DINO dual threshold sweep. Candidate features should include pose
diff/ratio/ranks, same-hand/target indicators, video calibration features, and
local cluster context. No submission artifact should be generated from the
pure-rival all-proposal source.

## Supervised Fix-vs-Break Gate Smoke

Added:

```text
tools/evaluate_fighter_rival_proposal_gate.py
```

The evaluator builds the high-recall rival proposal table, labels each proposal
as `fix`, `break`, `time_only`, or `unmatched` under local metric matching, and
trains leave-fight-out HGB/logreg gates to predict `fix`. It then scores
top-N/probability-selected fighter flips.

Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_fighter_rival_proposal_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --window 8 \
  --match-mode same_hand \
  --ratio 1.6 \
  --min-rival-score 0 \
  --models hgb,logreg \
  --output-summary data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_20260523_summary.csv \
  --output-proposals data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_20260523_proposals.csv \
  --top-k 40
```

Proposal table:

```text
proposals=174
fix=41
break=130
time_only=3
```

Best OOF selected row is still negative:

```text
model=logreg
threshold_type=top_n
top_n=1
selected=1
fix=0
break=1
fighter_delta=-0.000455
macro_delta=-0.000091
time_delta=+0.000000
fp_penalty_delta=+0.000000
```

Best HGB row is also negative:

```text
model=hgb
threshold_type=top_n
top_n=1
selected=1
fix=0
break=1
fighter_delta=-0.000466
macro_delta=-0.000093
time_delta=+0.000000
fp_penalty_delta=+0.000000
```

OOF probability diagnostics show no useful separation:

```text
hgb    fix_mean=0.194215  break_mean=0.228863
logreg fix_mean=0.432471  break_mean=0.414434
```

Decision: killed for the current feature set. The broad rival proposals still
have pass-level fix-only headroom, but pose score, row metadata, and simple
local proposal counts do not predict fix-vs-break well enough. The next
fighter gate needs a stronger evidence source, likely visual/tracklet
consistency or candidate-pool temporal identity features, before another
submission artifact is justified.

## Appearance/Dual Sidecar Follow-Up

Extended `tools/evaluate_fighter_rival_proposal_gate.py` with optional sidecar
features:

```text
--appearance-summary
--dual-candidate
```

These add per-video color/embedding calibration fields and an aligned DINO
dual-fighter flip indicator to the fix-vs-break model.

Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_fighter_rival_proposal_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --window 8 \
  --match-mode same_hand \
  --ratio 1.6 \
  --min-rival-score 0 \
  --appearance-summary data/processed/appearance_audits/dino_color_strict_clearens094012_train_yolo11s_20260522/train_fighter_appearance_video_summary.csv \
  --dual-candidate data/processed/vit_features/dinov2_vits14_dual_fighter_clip8_stride2_strict_clearens094012_margin098_rows_20260522.csv \
  --models hgb,logreg \
  --output-summary data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_appdual_20260523_summary.csv \
  --output-proposals data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_appdual_20260523_proposals.csv \
  --top-k 40
```

Best OOF selected row remains negative:

```text
model=logreg
threshold_type=top_n
top_n=5
selected=5
fix=2
break=3
fighter_delta=-0.000414
macro_delta=-0.000083
time_delta=+0.000000
fp_penalty_delta=+0.000000
```

Probability separation remains weak/wrong:

```text
hgb    fix_mean=0.226952  break_mean=0.255967
logreg fix_mean=0.457822  break_mean=0.453227
```

Decision: also killed. The existing per-video appearance summary and DINO dual
sidecar are not enough to gate broad rival proposals. The next fighter attempt
needs row-level visual/tracklet evidence, not only video-level calibration
metadata.

## Pose-Feature Gate Follow-Up

Added self/rival pose-candidate geometry features to the supervised gate:

```text
closing
arm_forward
reach
proximity
conf
role_conf
attacker_red_score
attacker_blue_score
attacker_color_margin
target_dist_norm
head_dist_norm
body_dist_norm
forward_norm
```

For each feature the model receives the self value, rival value, rival-self
delta, and rival/self ratio. Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_fighter_rival_proposal_gate.py \
  --base data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --window 8 \
  --match-mode same_hand \
  --ratio 1.6 \
  --min-rival-score 0 \
  --models hgb,logreg \
  --output-summary data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_posefeat_20260523_summary.csv \
  --output-proposals data/processed/diagnostics/fighter_rival_fixbreak_gate_train_yolo11s_w8_samehand_r16_posefeat_20260523_proposals.csv \
  --top-k 40
```

The pose features do not improve the gate:

```text
best pose-feature row:
model=hgb
threshold_type=top_n
top_n=12
selected=12
fix=1
break=11
fighter_delta=+0.000084
macro_delta=-0.000019
time_delta=+0.000000
fp_penalty_delta=+0.000000
min_video_delta=-0.001575

first HGB top1:
fix=0
break=1
fighter_delta=-0.000440
macro_delta=-0.000088

best logreg active rows are also negative or no-op.
```

Decision: killed. Adding row-level pose geometry from the candidate scorer is
not enough to separate fixes from breaks. The next fighter path still needs a
different source of identity evidence, such as visual tracklet continuity or
manual/VLM row inspection of the high-recall proposal table.

## YOLO26X Narrow Rival Test Artifact

The coverage sweep also exposed a low-churn direct rival rule that is not the
high-recall noisy source:

```text
source=val_yolo26x_conf035
window=0
match_mode=same_hand_target
ratio=1.6
min_rival_score=0.8
proposed=11
fix=4
break=7
all_fighter_delta=+0.001106
all_macro_delta=+0.000221
all_time_delta=+0.000000
all_fp_penalty_delta=+0.000000
all_min_video_delta=-0.001266
```

This does not pass the fighter component gate, but it is a small independent
fighter-only micro-signal. I materialized the matching test rule on the current
blocked-only public anchor:

```bash
.venv/bin/python tools/make_fighter_rival_flip_submission.py \
  --input submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --window 0 \
  --match-mode same_hand_target \
  --ratio 1.6 \
  --min-rival-score 0.8 \
  --update-mode fighter_only \
  --output submissions/prime_publicbest_nonhack_blockedonly_yolo26x_rival_fighter_w0_samehandtarget_r16_min08_fighteronly_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_yolo26x_rival_fighter_w0_samehandtarget_r16_min08_fighteronly_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Test diff versus the live blocked-only public anchor:

```text
changed_rows=5
changed_by_column: fighter=5
by_video: agn_047=1, agn_049=2, agn_063=1, agn_064=1
by_effectiveness: landed=3, miss=2
transitions: blue->red=4, red->blue=1
```

Diff and visual artifacts:

```text
data/processed/diagnostics/yolo26x_rival_fighter_w0_samehandtarget_r16_min08_fighteronly_testdiff_20260523.csv
data/processed/visual_audits/fighter_rival_yolo26x_20260523/w0_samehandtarget_r16_min08_fighteronly_rows.csv
data/processed/visual_audits/fighter_rival_yolo26x_20260523/w0_samehandtarget_r16_min08_fighteronly_contact_sheet.jpg
```

Manual visual read: the five flips are mixed. Some candidate fighters are
plausible, but several frames are ambiguous or still look compatible with the
original fighter label. Keep this as a validated no-upload micro-candidate, not
as an automatic leaderboard candidate.
