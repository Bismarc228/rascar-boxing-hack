# Root-Out Ptype-Only Candidate - 2026-05-23

Goal: materialize the revived root-out `punch_type_margin_0.1` component pass
as test-side artifacts without touching timing, fighter, hand, target,
effectiveness, clear flags, ids, or row count.

No Kaggle upload was made.

## Local Evidence

From `notes/punch_type_component_worker_2026-05-23.md`, the matching root-out
component pass is:

```text
rootout source_before_pose_logreg              macro=0.294998 ptype=0.175549
rootout pose_logreg_punch_type_margin_0p1      macro=0.299664 ptype=0.222214
delta                                          macro=+0.004666 ptype=+0.046665
jab_delta=+0.126635 uppercut_delta=+0.167722 min_root_ptype_delta=+0.028773
```

All non-`punch_type` components are fixed in this local comparison.

The same ptype signal does not transfer cleanly to the strict/current-anchor
OOF row source:

```text
strict pose_logreg_m01_frame_fighter_hand_on_anchor
macro_delta=+0.000576
ptype_delta=+0.005761
jab_delta=-0.016575
uppercut_delta=-0.005376
min_root_ptype_delta=-0.013645
missing_exact_source_rows=942
```

So the direct root-out stack candidate is the faithful artifact. The current
public-anchor splice below is diagnostic only.

## Root-Out Stack Artifact

Command:

```bash
python3 tools/combine_submission_columns.py \
  --base submissions/prime_rootout_pose_gate035_geom_id_splice_047_062_063_protected037038039_OFFLINE_CANDIDATE.csv \
  --source punch_type=submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_eff_p01_e005_splice_047_062_063_protected037038039_OFFLINE_CANDIDATE.csv \
  --output submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_p01_only_splice_047_062_063_protected037038039_20260523_OFFLINE_CANDIDATE.csv

python3 tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_p01_only_splice_047_062_063_protected037038039_20260523_OFFLINE_CANDIDATE.csv
```

Result:

```text
changed_by_column=punch_type:231
Validation passed.
```

Diff versus root-out protected geom-id base:

```text
changed_clear_rows=231
by_video=agn_047:80,agn_062:74,agn_063:77
transitions:
  cross->jab=86
  hook->jab=56
  hook->uppercut=40
  cross->hook=21
  cross->uppercut=19
  hook->cross=9
```

## Current Public-Anchor Diagnostic

Command:

```bash
python3 tools/combine_submission_columns.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --source punch_type=submissions/yolo11s_fighterhand_thr115_same10_cross4_rootcount1_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_eff_p01_e005_OFFLINE_CANDIDATE.csv \
  --output data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_allvideo_intermediate_20260523.csv

python3 tools/make_submission_diff_ablation.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --override data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_allvideo_intermediate_20260523.csv \
  --video-keys agn_047,agn_062,agn_063 \
  --output submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty

python3 tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_20260523_OFFLINE_CANDIDATE.csv
```

Result:

```text
selected=625
base_clear=696
output_clear=696
selected_clear_delta=+0
Validation passed.
```

Clear-row diff:

```text
changed_clear_rows=233
by_video=agn_047:89,agn_062:78,agn_063:66
```

This diagnostic is broad and not well-supported by strict OOF transfer. Do not
upload it unless a later guard finds a much narrower subset.

## Strict OOF Transition Guard

The full strict/current-anchor aligned root-out transfer was weak:

```text
all_pose_logreg n=130 macro_delta=+0.000576 ptype_delta=+0.005761
minroot_ptype=-0.013645 wins=8 losses=5
```

A simple transition guard that only allows rows whose base `punch_type` is
`cross` or `hook` keeps most of the ptype gain and removes the negative
root-level ptype tail:

```text
base_cross_hook_only n=63 macro_delta=+0.000519 ptype_delta=+0.005187
minroot_ptype=+0.003021 wins=8 losses=4
```

Individual transition checks on strict OOF:

```text
cross->jab       n=8  macro_delta=+0.000221 ptype_delta=+0.002212 minroot=+0.000000
hook->cross      n=31 macro_delta=+0.000347 ptype_delta=+0.003473 minroot=+0.000000
cross->hook      n=9  macro_delta=+0.000120 ptype_delta=+0.001200 minroot=-0.003663
hook->jab        n=6  macro_delta=-0.000079 ptype_delta=-0.000789 minroot=-0.001035
hook->uppercut   n=4  macro_delta=-0.000145 ptype_delta=-0.001455 minroot=-0.002701
uppercut->cross  n=9  macro_delta=-0.000441 ptype_delta=-0.004412 minroot=-0.016667
```

This does not make the current-anchor transfer a strong upload candidate, but
it gives a safer diagnostic than the all-transition splice.

## Current Public-Anchor Guarded Diagnostic

Command:

```bash
python3 tools/filter_row_updates.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --candidate data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_allvideo_intermediate_20260523.csv \
  --copy-columns punch_type \
  --base-filter punch_type=cross,hook \
  --output data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_base_crosshook_intermediate_20260523.csv \
  --fail-if-empty

python3 tools/make_submission_diff_ablation.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --override data/processed/diagnostics/prime_publicbest_nonhack_rootout_logreg_noctx_c100_ptype_p01_base_crosshook_intermediate_20260523.csv \
  --video-keys agn_047,agn_062,agn_063 \
  --output submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_20260523_OFFLINE_CANDIDATE.csv \
  --fail-if-empty

python3 tools/validate_data.py \
  --data-root data/raw \
  --submission submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_20260523_OFFLINE_CANDIDATE.csv
```

Result:

```text
selected=373
base_clear=696
output_clear=696
selected_clear_delta=+0
Validation passed.
```

Clear-row diff:

```text
changed_clear_rows=176
by_video=agn_047:68,agn_062:60,agn_063:48
transitions:
  hook->jab=97
  cross->jab=69
  hook->uppercut=64
  cross->uppercut=52
  cross->hook=51
  hook->cross=40
```

The first guarded diagnostic also copied ptype values into clear=false rows.
A cleaner clear-only version was generated with the same clear-row effect:

```text
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_clearonly_20260523_OFFLINE_CANDIDATE.csv
```

It validates and changes only the same `176` clear rows:

```text
by_video=agn_047:68,agn_062:60,agn_063:48
by_effectiveness=blocked:26,landed:103,miss:47
```

## Transition-Optimized Guard

The coarse base-`cross/hook` guard still carried weak/negative transitions.
`tools/filter_row_updates.py` now supports explicit transition filters:

```text
--transition-filter punch_type:before->after,before2->after2,...
```

Strict OOF transition-set search found two better clear fixed-row ptype
guards:

```text
transitionopt
transitions=cross->hook,cross->jab,cross->uppercut,hook->cross,jab->cross,jab->hook,uppercut->hook
n=110
macro=0.416503 delta=+0.001242
punch_type=0.224133 delta=+0.012417

transitionopt_minroot
transitions=cross->hook,cross->jab,hook->cross,jab->cross,jab->hook,uppercut->hook
n=105
macro=0.416448 delta=+0.001187
punch_type=0.223587 delta=+0.011871
```

Both are stronger than the earlier base-`cross/hook` strict OOF guard
(`macro +0.000519`, `punch_type +0.005187`).

Current-anchor clear-only core test artifacts:

```text
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitionopt_clearonly_20260523_OFFLINE_CANDIDATE.csv
selected=131
by_video=agn_047:58,agn_062:34,agn_063:39
by_effectiveness=blocked:35,landed:53,miss:43
Validation passed.

submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitionopt_minroot_clearonly_20260523_OFFLINE_CANDIDATE.csv
selected=121
by_video=agn_047:56,agn_062:31,agn_063:34
by_effectiveness=blocked:35,landed:48,miss:38
Validation passed.
```

The `transitionopt` artifact is the best local macro ptype diagnostic; the
`transitionopt_minroot` artifact is the lower-churn conservative sibling.

## Compact Transition Subsets

An exhaustive strict OOF subset sweep over the seven `transitionopt`
transitions was saved to:

```text
data/processed/diagnostics/punch_type_transition_subset_sweep_20260523.csv
```

Two lower-churn subsets are useful private-risk diagnostics:

```text
transitioncompact75
transitions=cross->hook,cross->jab,cross->uppercut,hook->cross,jab->hook,uppercut->hook
n=75
macro_delta=+0.001058
punch_type_delta=+0.010584
min_root_punch_type_delta=+0.003021

transitioncompact70
transitions=cross->hook,cross->jab,hook->cross,jab->hook,uppercut->hook
n=70
macro_delta=+0.001004
punch_type_delta=+0.010038
min_root_punch_type_delta=+0.005909
```

`transitioncompact70` is the better risk/churn tradeoff: it keeps most of the
`transitionopt_minroot` component gain (`punch_type_delta=+0.011871`) while
dropping the OOF ptype-change count from `105` to `70`.

Validated current-anchor clear-only core test artifacts:

```text
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitioncompact75_clearonly_20260523_OFFLINE_CANDIDATE.csv
selected=121
by_video=agn_047:57,agn_062:27,agn_063:37
by_effectiveness=blocked:30,landed:51,miss:40
Validation passed.

submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitioncompact70_clearonly_20260523_OFFLINE_CANDIDATE.csv
selected=111
by_video=agn_047:55,agn_062:24,agn_063:32
by_effectiveness=blocked:30,landed:46,miss:35
Validation passed.
```

Neither compact artifact was uploaded.

## Decision

Keep the root-out stack ptype-only artifact as the faithful materialization of
the component pass:

```text
submissions/prime_rootout_pose_gate035_geom_id_logreg_noctx_c100_ptype_p01_only_splice_047_062_063_protected037038039_20260523_OFFLINE_CANDIDATE.csv
```

The current-anchor diagnostic is useful for diff review only:

```text
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_basecrosshook_clearonly_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitionopt_clearonly_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitionopt_minroot_clearonly_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitioncompact75_clearonly_20260523_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_blockedonly_rootout_logreg_noctx_c100_ptype_p01_core047062063_transitioncompact70_clearonly_20260523_OFFLINE_CANDIDATE.csv
```
