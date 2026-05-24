# Clear Precision-Ranked Gate Worker - 2026-05-23

Goal: check revived hypothesis #2, a precision-ranked fixed-row clear/FP gate.
No Kaggle upload was made. Only OOF clear-row drops were evaluated; frame,
fighter, hand, target, punch_type, and effectiveness were not changed.

One requested context file was not present in this worktree:

```text
notes/combined_fixed_row_clear_gate_2026-05-22.md
```

## Tool

Added:

```text
tools/evaluate_clear_precision_ranked_gate.py
```

The evaluator maps the existing `candidate_token_clear_*` `p_keep` diagnostics
onto the current strict anchor by exact row key first and relaxed
`video_key,frame,fighter,hand,target` key second. It then scores top-K clear
drops under four guards:

```text
blocked_miss
blocked
miss
all
```

It evaluates raw low-`p_keep` rankers plus a lightweight leave-fight-out HGB
precision ranker trained only on cached p_keep/row metadata. No feature model
was rebuilt.

## Command

```bash
python3 tools/evaluate_clear_precision_ranked_gate.py --top-k 30
```

Outputs:

```text
data/processed/diagnostics/clear_precision_ranked_thresholds_20260523.csv
data/processed/diagnostics/clear_precision_ranked_video_20260523.csv
data/processed/diagnostics/clear_precision_ranked_rows_20260523.csv
```

## Anchor

Source:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Baseline:

```text
macro=0.415261
time=0.530111
fighter=0.537953
fp_penalty=0.061715
rows=1188
labels: tp_scorable=1044, fp=92, tp_time_only=52
```

Diagnostic cache match coverage:

```text
pose_audio_matched exact=632 relaxed=552 missing=4 duplicate_relaxed=1
pose_audio_rgb     exact=632 relaxed=552 missing=4 duplicate_relaxed=1
smoke_debug        exact=632 relaxed=552 missing=4 duplicate_relaxed=1
```

## Best Thresholds By Guard

Best row by macro for each guard/ranker:

| model | guard | top_k | macro | delta | fp_delta | dropped fp/scorable/time | fp/scorable | min video delta |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| oof_hgb_fp_pkeep | miss | 8 | 0.418058 | +0.002797 | -0.004533 | 4/3/1 | 1.333333 | +0.000000 |
| oof_hgb_fp_pkeep | all | 2 | 0.416996 | +0.001735 | -0.002029 | 1/1/0 | 1.000000 | +0.000000 |
| oof_hgb_fp_pkeep | blocked_miss | 1 | 0.415966 | +0.000705 | -0.000999 | 0/1/0 | 0.000000 | +0.000000 |
| oof_hgb_fp_pkeep | blocked | 1 | 0.414674 | -0.000588 | +0.000000 | 0/1/0 | 0.000000 | -0.007638 |
| pkeep_pose_audio_matched | blocked | 3 | 0.416335 | +0.001074 | -0.000449 | 1/1/1 | 1.000000 | +0.000000 |
| pkeep_pose_audio_matched | blocked_miss | 4 | 0.415968 | +0.000707 | -0.000449 | 1/2/1 | 0.500000 | +0.000000 |
| pkeep_pose_audio_matched | miss | 4 | 0.415433 | +0.000172 | -0.000999 | 1/2/1 | 0.500000 | -0.006273 |
| pkeep_pose_audio_matched | all | 6 | 0.415149 | -0.000112 | -0.000449 | 1/4/1 | 0.250000 | -0.005574 |

Rows satisfying macro >= +0.002 and FP/scorable > 1.0:

| model | guard | top_k | macro | delta | fp_delta | dropped fp/scorable/time | min video delta |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| oof_hgb_fp_pkeep | miss | 8 | 0.418058 | +0.002797 | -0.004533 | 4/3/1 | +0.000000 |

Rows satisfying fp_delta <= -0.006 all lost macro. The best such row was:

| model | guard | top_k | macro | delta | fp_delta | dropped fp/scorable/time | fp/scorable | min video delta |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| oof_hgb_fp_pkeep | miss | 18 | 0.414183 | -0.001079 | -0.006038 | 5/12/1 | 0.416667 | -0.041334 |

## Video Detail For Best Candidate

Best candidate:

```text
model=oof_hgb_fp_pkeep
guard=miss
top_k=8
macro=0.418058
delta=+0.002797
fp_penalty=0.057182
fp_delta=-0.004533
dropped_fp=4
dropped_tp_scorable=3
dropped_tp_time_only=1
```

Affected videos:

| video | final delta | fp delta | time delta | fighter delta | dropped fp/scorable/time |
| --- | ---: | ---: | ---: | ---: | --- |
| agn_003 | +0.036091 | -0.040199 | -0.007407 | +0.000000 | 2/1/0 |
| agn_010 | +0.000270 | -0.018733 | -0.021399 | -0.012346 | 2/2/1 |

No video collapsed for the best candidate, but the gain is concentrated in only
two videos.

## Decision

Strict pass criteria:

```text
fp_penalty -0.006 or better: FAIL (-0.004533 best with positive macro)
macro +0.002: PASS (+0.002797)
dropped_fp / dropped_tp_scorable > 1.0: PASS (4/3 = 1.333333)
no video collapse: PASS on best candidate
```

Verdict: fail the strict pass gate. The revived precision-ranked shape is not
dead in the weakest sense, because miss-only top-K ranking finds a real OOF
macro lift, but it does not remove enough FP before the scorable TP loss starts
to dominate. Broader blocked/miss and all-row controls do not meet the macro
threshold, and thresholds that reach the requested FP-penalty improvement
regress macro with poor FP/scorable precision.

## Local-Feature Ranker Follow-Up

Added an optional OOF ranker mode:

```text
tools/evaluate_clear_precision_ranked_gate.py --ranker-feature-set local
```

This preserves the old default `base` ranker and adds selected-row neighborhood
features for the HGB ranker only:

- counts of nearby selected rows in +/-5, +/-10, +/-15, +/-30, and +/-60 frames;
- same-fighter, opposite-fighter, same-fighter+hand, same-target, same-type,
  and same-effectiveness counts;
- local landed/blocked/miss counts;
- nearest-neighbor distances for any row, same fighter, same fighter+hand,
  opposite fighter, and same effectiveness.

Command:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --ranker-feature-set local \
  --top-k 40 \
  --output-thresholds data/processed/diagnostics/clear_precision_ranked_local_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_precision_ranked_local_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_precision_ranked_local_rows_20260523.csv
```

Result: killed. The local OOF HGB ranker is worse than the previous base OOF HGB
ranker.

Best old base OOF HGB row remains:

```text
guard=miss
top_k=8
macro_delta=+0.002797
time_delta=-0.002216
fighter_delta=-0.000950
fp_penalty_delta=-0.004533
dropped_fp=4
dropped_tp_scorable=3
dropped_tp_time_only=1
```

Best local OOF HGB row by macro:

```text
guard=miss/all
top_k=1
macro_delta=-0.000145
time_delta=+0.000063
fighter_delta=-0.000950
fp_penalty_delta=+0.000000
dropped_fp=0
dropped_tp_scorable=1
dropped_tp_time_only=0
```

First local OOF HGB row that actually drops an FP:

```text
guard=miss
top_k=6
macro_delta=-0.001274
time_delta=-0.001963
fighter_delta=-0.002849
fp_penalty_delta=-0.000713
dropped_fp=1
dropped_tp_scorable=4
dropped_tp_time_only=1
```

The local ranker top miss rows begin with true matched evidence rather than FP
rows, so the added neighborhood features overfit fold/context structure instead
of improving FP precision. This is not a candidate path; keep only as a
negative-control evaluator mode.

## Metadata-Only Test-Side Control

Added a reproducible no-`p_keep` control mode:

```text
tools/evaluate_clear_precision_ranked_gate.py --no-default-pkeep-source
```

This allows the OOF HGB FP ranker to train only on selected-row metadata
(`frame`, video metadata, row labels) plus optional local selected-row
neighborhood features. The purpose is to check whether the old `miss top-8`
OOF signal can be replaced by a test-side feasible ranker that does not need
the missing `candidate_token_clear_*` test cache.

Commands:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --no-default-pkeep-source \
  --ranker-feature-set base \
  --top-k 40 \
  --output-thresholds data/processed/diagnostics/clear_precision_ranked_metadata_base_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_precision_ranked_metadata_base_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_precision_ranked_metadata_base_rows_20260523.csv

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --no-default-pkeep-source \
  --ranker-feature-set local \
  --top-k 40 \
  --output-thresholds data/processed/diagnostics/clear_precision_ranked_metadata_local_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_precision_ranked_metadata_local_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_precision_ranked_metadata_local_rows_20260523.csv
```

Best metadata-only base row:

```text
guard=all
top_k=1
macro_delta=+0.001021
time_delta=+0.000000
fighter_delta=+0.000000
fp_penalty_delta=-0.000999
dropped_fp=1
dropped_tp_scorable=0
dropped_tp_time_only=0
affected_video=agn_003
```

The one recovered FP is a `landed` row, so it is not a usable non-landed
clear/FP policy. The best guarded non-landed row is already negative:

```text
guard=miss
top_k=3
macro_delta=-0.000711
time_delta=-0.001994
fighter_delta=-0.001425
fp_penalty_delta=-0.000713
dropped_fp=1
dropped_tp_scorable=2
dropped_tp_time_only=0
```

Metadata-only local features are worse:

```text
best macro row:
  macro_delta=-0.000145
  time_delta=+0.000063
  fighter_delta=-0.000950
  fp_penalty_delta=+0.000000
  dropped_fp=0
first FP drop:
  guard=miss
  top_k=4
  macro_delta=-0.001220
  time_delta=-0.001931
  fighter_delta=-0.004274
  fp_penalty_delta=-0.000713
  dropped_fp=1
  dropped_tp_scorable=3
```

Decision: killed as a reproducible test-side replacement for the OOF-only
`p_keep` ranker. The old `p_keep` `miss top-8` result remains the better
component diagnostic (`fp_penalty_delta=-0.004533`, `macro_delta=+0.002797`),
but it still lacks the required test-side cache/generator and should not be
materialized by a metadata-only proxy.

## Direct Strict-Current p_keep Ranked Recheck

After a train-all test bridge was added, the direct current-anchor
`candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv`
cache was rechecked as a top-K ranker:

```text
source=strict_current exact=1188 relaxed=0 missing=0 duplicate_relaxed=0
```

Best rows:

```text
guard=blocked_miss top7:
macro_delta=+0.003761
time_delta=+0.000075
fighter_delta=+0.001751
fp_penalty_delta=-0.003110
dropped_fp=3
dropped_tp_scorable=2
dropped_tp_time_only=2
min_video_delta=-0.000291

guard=blocked_miss top4:
macro_delta=+0.003255
time_delta=+0.000680
fighter_delta=+0.002357
fp_penalty_delta=-0.002080
dropped_fp=2
dropped_tp_scorable=1
dropped_tp_time_only=1
min_video_delta=+0.000000
```

This does not revive raw probability thresholding, but it changes the clear
follow-up shape: ranked top-K direct strict-current `p_keep` is now a
materialized low-churn candidate family. See
`notes/candidate_token_clear_trainall_test_2026-05-23.md`.
