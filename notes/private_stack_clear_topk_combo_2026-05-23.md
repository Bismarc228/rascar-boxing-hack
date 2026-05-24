# Private Stack + Candidate-Token Clear Top-K - 2026-05-23

Goal: test whether the materialized candidate-token clear top-K signal stacks
with the existing private component stack. GPU 1 was used only for the small
train-all token-clear inference runs. No Kaggle upload was made.

## OOF Stack Recheck

The clear ranker was evaluated on top of the strict OOF private stacks, using
the direct current-anchor `p_keep` cache:

```bash
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --predictions data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv \
  --no-default-pkeep-source \
  --pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --thresholds 1,2,3,4,5,6,7,8,9,10,12,15 \
  --top-k 12 \
  --output-thresholds data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_rows_20260523.csv
```

`transitionopt` stack baseline:

```text
macro=0.418474
time=0.530111
fighter=0.537953
fp_penalty=0.061715
```

Best added clear row:

```text
guard=blocked_miss
top_k=8
macro=0.420768
macro_delta_vs_stack=+0.002293
time_delta_vs_stack=-0.000785
fighter_delta_vs_stack=+0.000830
fp_penalty_delta_vs_stack=-0.002531
dropped_fp=2
dropped_tp_scorable=4
dropped_tp_time_only=2
min_video_delta=-0.005093
```

This gives a total strict OOF macro score of `0.420768` versus the strict base
`0.415261`, i.e. `total_macro_delta=+0.005507`. The improvement is not purely
additive: the added clear gate improves `fp_penalty` but slightly hurts `time`.

The `transitionopt_minroot` stack has the same added clear deltas and lands at
`macro=0.420713`.

Compact `transitioncompact70` was reconstructed as a saved strict OOF stack
after the first pass:

```bash
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --predictions data/processed/vit_features/component_stack_dinom03_ptypetransitioncompact70_targetguard_strict_oof_20260523.csv \
  --no-default-pkeep-source \
  --pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --thresholds 1,2,3,4,5,6,7,8,9,10,12,15 \
  --top-k 12 \
  --output-thresholds data/processed/diagnostics/clear_ranked_on_private_stack_transitioncompact70_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_ranked_on_private_stack_transitioncompact70_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_ranked_on_private_stack_transitioncompact70_rows_20260523.csv
```

`transitioncompact70` stack baseline:

```text
macro=0.418237
time=0.530111
fighter=0.537953
fp_penalty=0.061715
```

Best added clear row:

```text
guard=blocked_miss
top_k=8
macro=0.420530
macro_delta_vs_stack=+0.002293
time_delta_vs_stack=-0.000785
fighter_delta_vs_stack=+0.000830
fp_penalty_delta_vs_stack=-0.002531
dropped_fp=2
dropped_tp_scorable=4
dropped_tp_time_only=2
min_video_delta=-0.005093
```

Total strict OOF macro is `0.420530` versus strict base `0.415261`,
`total_macro_delta=+0.005269`. The added clear leg has the same component
movement as on `transitionopt`; the lower total score comes from the smaller
ptype leg.

## Test Artifacts

Unprotected top8:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 8 \
  --drop-effectiveness blocked,miss \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Protected top8 excluding `agn_037/038`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 8 \
  --drop-effectiveness blocked,miss \
  --protect-video-keys agn_037,agn_038 \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_noagn037038_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv
```

Both submissions passed `tools/validate_data.py`.

Unprotected diff versus its private-stack parent:

```text
changed_clear=8
by_video: agn_037=1, agn_038=1, agn_047=6
by_effectiveness: blocked=2, miss=6
```

Protected diff versus its private-stack parent:

```text
changed_clear=8
by_video: agn_047=8
by_effectiveness: blocked=3, miss=5
```

Diff versus the live blocked-only public anchor:

```text
unprotected changed_rows=187
changed_by_column: clear=8, punch_type=131, effectiveness=59, target=3

protected changed_rows=186
changed_by_column: clear=8, punch_type=131, effectiveness=59, target=3
```

Protected compact70 top8:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 8 \
  --drop-effectiveness blocked,miss \
  --protect-video-keys agn_037,agn_038 \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitioncompact70_yolo26l_pose_audio_e30_top8_blockedmiss_noagn037038_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Protected compact70 diff versus its private-stack parent:

```text
changed_rows=8
changed_by_column: clear=8
by_video: agn_047=8
by_effectiveness: blocked=3, miss=5
```

Protected compact70 diff versus the live blocked-only public anchor:

```text
changed_rows=168
changed_by_column: clear=8, punch_type=111, effectiveness=59, target=3
by_video: agn_047=68, agn_048=8, agn_049=4, agn_062=35, agn_063=43, agn_064=10
```

Diff audits:

```text
data/processed/diagnostics/private_stack_transitioncompact70_clear_top8_noagn037038_parentdiff_20260523.csv
data/processed/diagnostics/private_stack_transitioncompact70_clear_top8_noagn037038_vs_live_diff_20260523.csv
```

## HGB Pkeep Miss-Top1 Control

The narrow-guard audit found that `blocked`-only rows did not improve the FP
component on the private stacks. The only clean OOF row was the HGB ranker over
`p_keep` and metadata, guarded to `miss` and `top_k=1`:

```text
guard=miss
top_k=1
model=oof_hgb_fp_pkeep
macro_delta_vs_stack=+0.000913
time_delta_vs_stack=+0.000000
fighter_delta_vs_stack=+0.000000
fp_penalty_delta_vs_stack=-0.000999
dropped_fp=1
dropped_tp_scorable=0
dropped_tp_time_only=0
min_video_delta=+0.000000
```

A train-all materializer was added for this ranker:

```text
tools/make_clear_precision_ranked_submission.py
```

Protected compact70 HGB miss-top1:

```bash
.venv/bin/python tools/make_clear_precision_ranked_submission.py \
  --train-predictions data/processed/vit_features/component_stack_dinom03_ptypetransitioncompact70_targetguard_strict_oof_20260523.csv \
  --test-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitioncompact70_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --train-pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --test-pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitioncompact70_yolo26l_pose_audio_e30_top8_blockedmiss_noagn037038_20260523.csv \
  --guard miss \
  --drop-top-k 1 \
  --protect-video-keys agn_037,agn_038 \
  --output-diagnostics data/processed/diagnostics/clear_hgbpkeep_trainall_private_stack_transitioncompact70_miss_top1_noagn037038_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_hgbpkeep_clear_miss_top1_noagn037038_20260523_OFFLINE_CANDIDATE.csv
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw --submission submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_hgbpkeep_clear_miss_top1_noagn037038_20260523_OFFLINE_CANDIDATE.csv
Validation passed.
```

Test diff versus parent:

```text
changed_rows=1
changed_by_column: clear=1
by_video: agn_047=1
by_effectiveness: miss=1
selected_id=348
selected_frame=2428
selected_label=red uppercut left head miss
```

Test diff versus live anchor:

```text
changed_rows=167
changed_by_column: clear=1, punch_type=111, effectiveness=59, target=3
by_video: agn_047=67, agn_048=8, agn_049=4, agn_062=35, agn_063=43, agn_064=10
```

Diff and visual artifacts:

```text
data/processed/diagnostics/private_stack_transitioncompact70_hgbpkeep_miss_top1_noagn037038_parentdiff_20260523.csv
data/processed/diagnostics/private_stack_transitioncompact70_hgbpkeep_miss_top1_noagn037038_vs_live_diff_20260523.csv
data/processed/visual_audits/private_stack_clear_topk_20260523/transitioncompact70_hgbpkeep_miss_top1_noagn037038_rows.csv
data/processed/visual_audits/private_stack_clear_topk_20260523/transitioncompact70_hgbpkeep_miss_top1_noagn037038_contact_sheet.jpg
```

## Visual Audit

Generated protected top8 visual artifacts:

```bash
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python tools/make_submission_drop_contact_sheet.py \
  --base submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv \
  --candidate submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv \
  --diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_noagn037038_20260523.csv \
  --output-dir data/processed/visual_audits/private_stack_clear_topk_20260523 \
  --prefix transitionopt_top8_blockedmiss_noagn037038 \
  --frame-offsets=-4,0,4
```

Artifacts:

```text
data/processed/visual_audits/private_stack_clear_topk_20260523/transitionopt_top8_blockedmiss_noagn037038_contact_sheet.jpg
data/processed/visual_audits/private_stack_clear_topk_20260523/transitionopt_top8_blockedmiss_noagn037038_rows.csv
```

Manual read: the protected clear drops avoid `agn_037/038`, but they are not
clean false positives. All eight are active `agn_047` exchange rows; several
look like plausible jabs/hooks/crosses or plausible misses/blocks. This is not
a visually safe FP-only gate.

The protected `transitioncompact70` top8 drops the same eight row ids as the
protected `transitionopt` top8, so the same visual-risk read applies. The lower
churn comes from the compact ptype leg, not from safer clear removal.

The HGB miss-top1 control selects only id `348` in `agn_047`, but the visual
read is still not clean FP removal: the fighters are in a close exchange, and
`red uppercut left miss` is plausible. This is safer on OOF components but still
not a clean auto-upload row.

## Decision

The stack+clear combo is a real local/private-risk candidate family:
`total_macro_delta=+0.005507` on strict OOF for `transitionopt + clear top8`.
However, the added clear leg has `time_delta_vs_stack=-0.000785`, touches
visually plausible punches, and the test candidate has high total churn from
the private stack legs. Keep both CSVs as prepared diagnostic/private-risk
artifacts requiring explicit user approval before any Kaggle upload.

The more conservative sibling is:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv
```

It has lower live-anchor churn (`changed_rows=168`) and total strict OOF
`macro=0.420530`, but it inherits the same clear-drop visual risk. It is also a
no-upload private-risk candidate unless explicitly approved.

The HGB miss-top1 sibling is lower-churn still:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitioncompact70_targetguard_hgbpkeep_clear_miss_top1_noagn037038_20260523_OFFLINE_CANDIDATE.csv
```

Its OOF support is component-clean (`fp_penalty_delta_vs_stack=-0.000999`,
`time_delta_vs_stack=+0.000000`, `fighter_delta_vs_stack=+0.000000`,
`macro_delta_vs_stack=+0.000913`), but the single selected test row is visually
plausible. Keep it as a diagnostic/private-risk artifact, not an upload.

## LB Result - 2026-05-23

The user explicitly approved uploading the strongest prepared candidate in this
family. Uploaded:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Validation before upload:

```text
Validation passed.
```

Public LB:

```text
public_score=0.19069
current_nonhack_anchor_public_score=0.19516
public_delta=-0.00447
```

This should not become the anchor. The local component evidence remains useful
for private-risk analysis (`total_strict_oof_macro_delta=+0.005507`), but the
public result confirms that this high-churn stack plus visually risky clear leg
does not transfer to the public split.
