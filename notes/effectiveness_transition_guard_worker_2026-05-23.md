# Effectiveness transition guard worker - 2026-05-23

Goal: verify revived hypothesis #3 on fixed strict OOF rows, changing only
`effectiveness` and preserving `clear/frame/fighter/hand/target/punch_type`.
No Kaggle upload was made. No new visual backbone or DINO full finetune was run;
the checks use existing frozen VideoMAE/DINO/visual OOF row outputs.

Note: `notes/videomae_effectiveness_head_tuning_2026-05-22.md` was not present
under that exact path in this checkout. Used the available VideoMAE notes and
OOF artifacts listed below.

## Tool

Added:

```text
tools/evaluate_effectiveness_transition_guard.py
```

The tool uses the strict anchor as the only source of fixed row columns, aligns
candidates by `id,video_key`, copies only candidate `effectiveness`, and scores
transition guards:

```text
landed_to_miss
landed_to_blocked
landed_to_nonlanded
blocked_changes
miss_changes
nonlanded_changes
blocked_miss_swap
miss_to_landed_control
blocked_to_landed_control
all_effectiveness_changes
```

It writes:

```text
data/processed/diagnostics/effectiveness_transition_guard_20260523_summary.csv
data/processed/diagnostics/effectiveness_transition_guard_20260523_per_video.csv
data/processed/diagnostics/effectiveness_transition_guard_20260523_transition_confusion.csv
data/processed/diagnostics/effectiveness_transition_guard_20260523_candidate_audit.csv
```

Pass-row materialization:

```text
data/processed/vit_features/effectiveness_transition_guard_pass_rows_20260523/
data/processed/diagnostics/effectiveness_transition_guard_pass_20260523_summary.csv
data/processed/diagnostics/effectiveness_transition_guard_pass_20260523_per_video.csv
data/processed/diagnostics/effectiveness_transition_guard_pass_20260523_transition_confusion.csv
data/processed/diagnostics/effectiveness_transition_guard_pass_20260523_candidate_audit.csv
```

## Main command

```text
.venv/bin/python tools/evaluate_effectiveness_transition_guard.py \
  --anchor data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --candidate temporal_mlp=data/processed/vit_features/videomae_temporalbins_mlp_eff_landed_to_nonlanded_m07_strict_clearens094012_rows_20260522.csv \
  --candidate videomae_d2res_m010=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_d2res_h192_cw08_focal15_m010_seed17_rows_20260522.csv \
  --candidate videomae_d2res_seq_target=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_d2res_h192_cw08_focal15_m010_seed17_seq_fighter_target_reranker_rows_20260522.csv \
  --candidate videomae_direct_h192=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_rows_20260522.csv \
  --candidate videomae_direct_fighter_target=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_direct_h192_w07_ls004_m03_fighter_target_m095_rows_20260522.csv \
  --candidate videomae_eff_m08=data/processed/vit_features/clip_ptype_m028_videomae_eff_m08_rows_20260522.csv \
  --candidate videomae_eff_m09=data/processed/vit_features/clip_ptype_m028_videomae_eff_m09_rows_20260522.csv \
  --candidate restore_landed=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_restore_landed_rows_20260522.csv \
  --candidate restore_landed_blocked=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_restore_landed_blocked_rows_20260522.csv \
  --candidate restore_all=data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_restore_all_rows_20260522.csv \
  --candidate dino_roi_landed_to_nonlanded=data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_eff_m05_landed_to_nonlanded_rows_20260522.csv \
  --candidate dino_roi_margin03=data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_eff_margin03_rows_20260522.csv \
  --candidate dino_roi_margin05=data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_eff_margin05_rows_20260522.csv \
  --candidate visual_consensus_same=data/processed/vit_features/visual_consensus_intersection_same_dino_roi_m05_videomae_temporal_m07_strict_clearens094012_rows_20260522.csv \
  --candidate visual_consensus_nonlanded_dino=data/processed/vit_features/visual_consensus_intersection_nonlanded_dino_dino_roi_m05_videomae_temporal_m07_strict_clearens094012_rows_20260522.csv \
  --candidate visual_consensus_nonlanded_vm=data/processed/vit_features/visual_consensus_intersection_nonlanded_vm_dino_roi_m05_videomae_temporal_m07_strict_clearens094012_rows_20260522.csv \
  --candidate visual_consensus_union_dino=data/processed/vit_features/visual_consensus_union_dino_priority_dino_roi_m05_videomae_temporal_m07_strict_clearens094012_rows_20260522.csv \
  --candidate visual_consensus_union_vm=data/processed/vit_features/visual_consensus_union_vm_priority_dino_roi_m05_videomae_temporal_m07_strict_clearens094012_rows_20260522.csv \
  --output-prefix data/processed/diagnostics/effectiveness_transition_guard_20260523
```

Baseline:

```text
strict anchor weighted_macro=0.415261
strict anchor score_effectiveness=0.289848
rows=1188 videos=13
```

Pass rule used from the assignment:

```text
score_effectiveness_delta >= +0.020
weighted_macro_delta >= +0.0015
wins >= 8/13
videos with delta_effectiveness < -0.02 <= 2
```

## Component table

| candidate | variant | changed_rows | weighted_macro_delta | score_effectiveness_delta | wins | losses | bad_effectiveness_videos | pass_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dino_roi_margin03 | landed_to_nonlanded | 188 | +0.001744 | +0.021803 | 8 | 4 | 2 | true |
| videomae_d2res_m010 | all_effectiveness_changes | 292 | +0.001613 | +0.020161 | 9 | 4 | 1 | true |
| videomae_d2res_seq_target | all_effectiveness_changes | 292 | +0.001613 | +0.020161 | 9 | 4 | 1 | true |
| dino_roi_margin03 | all_effectiveness_changes | 356 | +0.001302 | +0.016271 | 6 | 7 | 5 | false |
| videomae_d2res_m010 | landed_to_nonlanded | 133 | +0.001196 | +0.014946 | 8 | 4 | 0 | false |
| dino_roi_margin03 | landed_to_miss | 137 | +0.000900 | +0.011256 | 8 | 4 | 0 | false |
| dino_roi_margin03 | landed_to_blocked | 51 | +0.000844 | +0.010547 | 5 | 6 | 2 | false |
| videomae_d2res_m010 | landed_to_miss | 112 | +0.000669 | +0.008360 | 6 | 6 | 0 | false |
| videomae_d2res_m010 | landed_to_blocked | 21 | +0.000527 | +0.006587 | 5 | 1 | 0 | false |
| temporal_mlp | landed_to_miss | 245 | +0.000440 | +0.005495 | 6 | 7 | 2 | false |
| videomae_d2res_m010 | nonlanded_changes | 159 | +0.000417 | +0.005214 | 6 | 7 | 4 | false |
| temporal_mlp | landed_to_nonlanded | 261 | +0.000392 | +0.004899 | 6 | 7 | 2 | false |
| videomae_d2res_m010 | miss_to_landed_control | 53 | +0.000096 | +0.001203 | 8 | 4 | 0 | false |
| dino_roi_margin03 | miss_to_landed_control | 47 | -0.000424 | -0.005299 | 5 | 3 | 2 | false |
| dino_roi_margin03 | nonlanded_changes | 168 | -0.000443 | -0.005532 | 6 | 6 | 5 | false |

`dino_roi_margin03 landed_to_nonlanded` is the best strict transition guard and
passes all thresholds. `videomae_d2res_m010 all_effectiveness_changes` also
passes all thresholds, but its narrow `landed_to_nonlanded` slice does not reach
the required `+0.020` effectiveness delta. The strict temporal VideoMAE MLP
from the prior note remains a fail: `+0.004899` effectiveness and `+0.000392`
weighted macro.

## Transition confusion

Best narrow visual guard:

```text
dino_roi_margin03 landed_to_nonlanded
transitions=landed->miss:137,landed->blocked:51
matched/scored improvements:
  landed->miss with gt=miss:      +53
  landed->blocked with gt=blocked:+12
matched/scored losses:
  landed->miss with gt=landed:    -64
  landed->blocked with gt=landed: -25
unscored_or_unmatched: 17 rows
```

Passing VideoMAE d2res broad effectiveness guard:

```text
videomae_d2res_m010 all_effectiveness_changes
transitions=landed->miss:112,landed->blocked:21,blocked->landed:42,blocked->miss:41,miss->blocked:23,miss->landed:53
matched/scored improvements:
  landed->miss gt=miss:      +36
  landed->blocked gt=blocked:+5
  blocked->landed gt=landed: +24
  blocked->miss gt=miss:     +12
  miss->blocked gt=blocked:  +6
  miss->landed gt=landed:    +27
matched/scored losses:
  landed->miss gt=landed:    -53
  landed->blocked gt=landed: -9
  blocked->landed gt=blocked:-9
  blocked->miss gt=blocked:  -7
  miss->blocked gt=miss:     -11
  miss->landed gt=miss:      -13
```

Controls:

```text
dino_roi_margin03 miss_to_landed_control: -0.005299 effectiveness, -0.000424 weighted macro
videomae_d2res_m010 miss_to_landed_control: +0.001203 effectiveness, +0.000096 weighted macro
temporal_mlp miss_to_landed_control: 0 changes
```

The DINO control behaves as expected: broad nonlanded changes hurt and
miss->landed is negative. VideoMAE d2res has a small positive miss->landed
control, but it is far below pass thresholds and should not be treated as an
independent candidate.

## Per-video risk

`dino_roi_margin03 landed_to_nonlanded`:

| video_key | changed_rows | delta_weighted_macro | delta_effectiveness | bad_effectiveness_video |
| --- | --- | --- | --- | --- |
| agn_003 | 18 | -0.001445 | -0.018060 | false |
| agn_004 | 0 | +0.000000 | +0.000000 | false |
| agn_010 | 5 | -0.001818 | -0.022727 | true |
| agn_023 | 24 | +0.000519 | +0.006486 | false |
| agn_024 | 23 | +0.005133 | +0.064167 | false |
| agn_025 | 16 | +0.000120 | +0.001506 | false |
| agn_056 | 3 | -0.000246 | -0.003081 | false |
| agn_057 | 6 | +0.003141 | +0.039267 | false |
| agn_058 | 6 | -0.002133 | -0.026667 | true |
| agn_069 | 28 | +0.002596 | +0.032454 | false |
| agn_070 | 27 | +0.008020 | +0.100249 | false |
| agn_071 | 27 | +0.007992 | +0.099898 | false |
| agn_072 | 5 | +0.000796 | +0.009946 | false |

`videomae_d2res_m010 all_effectiveness_changes`:

| video_key | changed_rows | delta_weighted_macro | delta_effectiveness | bad_effectiveness_video |
| --- | --- | --- | --- | --- |
| agn_003 | 8 | +0.002972 | +0.037152 | false |
| agn_004 | 19 | -0.003533 | -0.044161 | true |
| agn_010 | 10 | +0.000393 | +0.004914 | false |
| agn_023 | 33 | -0.000306 | -0.003826 | false |
| agn_024 | 27 | +0.001886 | +0.023569 | false |
| agn_025 | 28 | -0.000120 | -0.001506 | false |
| agn_056 | 27 | +0.000847 | +0.010588 | false |
| agn_057 | 33 | -0.001144 | -0.014305 | false |
| agn_058 | 10 | +0.001156 | +0.014444 | false |
| agn_069 | 29 | +0.004038 | +0.050480 | false |
| agn_070 | 24 | +0.012194 | +0.152422 | false |
| agn_071 | 33 | +0.000244 | +0.003046 | false |
| agn_072 | 11 | +0.002342 | +0.029270 | false |

## Decision

Pass, with a scope caveat:

- Visual transition guard passes cleanly: `dino_roi_margin03 landed_to_nonlanded`
  gives `+0.021803` score_effectiveness, `+0.001744` weighted macro, `8/13`
  wins, and exactly `2` videos below `-0.02` effectiveness delta.
- VideoMAE d2res broad effectiveness replacement also passes: `+0.020161`
  score_effectiveness, `+0.001613` weighted macro, `9/13` wins, and `1` bad
  video. This pass is not the narrow landed->nonlanded guard; it needs the
  broader blocked/miss and miss->landed corrections.
- Strict temporal VideoMAE landed->nonlanded remains too weak and fails.

Candidate audit caveat: the old non-strict d2res source has 38 extra candidate
rows relative to the strict anchor and 4 fixed-column mismatches in the source
CSV. The guarded output itself keeps all fixed columns from the strict anchor.

## Validation

```text
.venv/bin/python -m py_compile tools/evaluate_effectiveness_transition_guard.py
git diff --check
```

Both passed.
