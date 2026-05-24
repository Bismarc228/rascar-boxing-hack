# Candidate-Token Clear Train-All Test Bridge - 2026-05-23

Goal: check whether the direct current-anchor candidate-token clear model can be
materialized on the test anchor. GPU 1 was used for the small Transformer head;
no Kaggle upload was made.

## Tool

Added:

```text
tools/make_candidate_token_clear_submission.py
tools/make_submission_drop_contact_sheet.py
```

The tool loads the historical OOF-only evaluator from:

```text
d80c86c38b12e4b3a0c05126daaa812b8dcaf46e:tools/evaluate_candidate_token_reranker.py
```

It keeps the legacy feature/model code intact and adds train-all inference:

- train on all clear rows from the strict current OOF anchor;
- infer `p_keep` for clear rows in the current public blocked-only test anchor;
- optionally set `clear=false` either by threshold or by lowest-`p_keep` top-K
  inside an effectiveness guard;
- write test diagnostics and a full fixed-row submission CSV.

The contact-sheet helper compares a base and candidate submission, finds
`clear=true -> clear=false` diffs, decodes frames with
`CUDA_VISIBLE_DEVICES=1 ffmpeg -hwaccel cuda -c:v hevc_cuvid`, and writes a
row CSV plus `-4/0/+4` frame contact sheet.

## OOF Top-K Recheck

The direct current-anchor `p_keep` cache was rechecked with the ranked clear
gate instead of only global probability thresholds:

```bash
.venv/bin/python tools/evaluate_clear_precision_ranked_gate.py \
  --no-default-pkeep-source \
  --pkeep-source strict_current=data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --thresholds 1,2,3,4,5,6,7,8,9,10,12,15,20,25,30 \
  --top-k 20 \
  --output-thresholds data/processed/diagnostics/clear_precision_ranked_strict_current_pkeep_thresholds_20260523.csv \
  --output-video data/processed/diagnostics/clear_precision_ranked_strict_current_pkeep_video_20260523.csv \
  --output-ranks data/processed/diagnostics/clear_precision_ranked_strict_current_pkeep_rows_20260523.csv
```

Best rows by local OOF macro:

```text
guard=blocked_miss top7:
fp_penalty_delta=-0.003110
time_delta=+0.000075
fighter_delta=+0.001751
macro_delta=+0.003761
dropped_fp=3
dropped_tp_scorable=2
dropped_tp_time_only=2
min_video_delta=-0.000291

guard=blocked_miss top4:
fp_penalty_delta=-0.002080
time_delta=+0.000680
fighter_delta=+0.002357
macro_delta=+0.003255
dropped_fp=2
dropped_tp_scorable=1
dropped_tp_time_only=1
min_video_delta=+0.000000
```

The old probability threshold `p_keep < 0.02` does not transfer as a threshold:
on test it drops no `miss` rows because the minimum test `miss` `p_keep` is
`0.022098919`. Top-K is the usable materialization shape.

## Test Artifacts

Train-all command shape:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_candidate_token_clear_submission.py \
  --epochs 30 \
  --feature-mode pose_audio \
  --drop-top-k 7 \
  --drop-effectiveness blocked,miss \
  --output-diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_yolo26l_pose_audio_e30_top7_blockedmiss_20260523.csv \
  --output-submission submissions/prime_publicbest_nonhack_blockedonly_candidate_token_clear_trainall_yolo26l_poseaudio_e30_top7_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

All generated submissions passed `tools/validate_data.py`.

### Top7 Blocked/Miss

```text
submission:
submissions/prime_publicbest_nonhack_blockedonly_candidate_token_clear_trainall_yolo26l_poseaudio_e30_top7_blockedmiss_20260523_OFFLINE_CANDIDATE.csv

diagnostics:
data/processed/diagnostics/candidate_token_clear_trainall_test_yolo26l_pose_audio_e30_top7_blockedmiss_20260523.csv

changed_ids=7
changed_column=clear only
by_video: agn_037=1, agn_038=2, agn_047=4
by_effectiveness: blocked=3, miss=4
```

Changed rows:

```text
id=7   agn_037 frame=3064 miss    p_keep=0.030294752
id=88  agn_038 frame=3375 blocked p_keep=0.031318933
id=150 agn_038 frame=5232 miss    p_keep=0.023793975
id=295 agn_047 frame=211  miss    p_keep=0.024667436
id=325 agn_047 frame=1030 blocked p_keep=0.009783677
id=326 agn_047 frame=1045 blocked p_keep=0.011750042
id=330 agn_047 frame=1171 miss    p_keep=0.022098919
```

### Top4 Blocked/Miss

```text
submission:
submissions/prime_publicbest_nonhack_blockedonly_candidate_token_clear_trainall_yolo26l_poseaudio_e30_top4_blockedmiss_20260523_OFFLINE_CANDIDATE.csv

diagnostics:
data/processed/diagnostics/candidate_token_clear_trainall_test_yolo26l_pose_audio_e30_top4_blockedmiss_20260523.csv

changed_ids=4
changed_column=clear only
by_video: agn_038=1, agn_047=3
by_effectiveness: blocked=2, miss=2
```

### Top4 Blocked/Miss No agn_037/038

```text
submission:
submissions/prime_publicbest_nonhack_blockedonly_candidate_token_clear_trainall_yolo26l_poseaudio_e30_top4_blockedmiss_noagn037038_20260523_OFFLINE_CANDIDATE.csv

diagnostics:
data/processed/diagnostics/candidate_token_clear_trainall_test_yolo26l_pose_audio_e30_top4_blockedmiss_noagn037038_20260523.csv

changed_ids=4
changed_column=clear only
by_video: agn_047=4
by_effectiveness: blocked=2, miss=2
```

## Decision

This revives candidate-token clear as a materialized low-churn FP candidate,
but not as an automatic upload. The local signal is component-correct for
`fp_penalty` and does not hurt `time` in the top4/top7 OOF rows, but the test
top7 touches public-sensitive `agn_037/038`, and the generator trains only on
the strict current OOF rows available for this anchor. Treat these CSVs as
prepared private-risk candidates requiring explicit user approval before any
Kaggle upload.

## Visual Risk Audit

Generated top7 visual artifacts:

```bash
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python tools/make_submission_drop_contact_sheet.py \
  --base submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --candidate submissions/prime_publicbest_nonhack_blockedonly_candidate_token_clear_trainall_yolo26l_poseaudio_e30_top7_blockedmiss_20260523_OFFLINE_CANDIDATE.csv \
  --diagnostics data/processed/diagnostics/candidate_token_clear_trainall_test_yolo26l_pose_audio_e30_top7_blockedmiss_20260523.csv \
  --output-dir data/processed/visual_audits/candidate_token_clear_topk_20260523 \
  --prefix top7_blockedmiss \
  --frame-offsets=-4,0,4
```

Artifacts:

```text
data/processed/visual_audits/candidate_token_clear_topk_20260523/top7_blockedmiss_contact_sheet.jpg
data/processed/visual_audits/candidate_token_clear_topk_20260523/top7_blockedmiss_rows.csv
```

Manual visual read:

| id | video | label | Visual read | Risk |
| ---: | --- | --- | --- | --- |
| 7 | `agn_037` | red hook miss | Active close exchange; red action is plausible but not a clean scoring moment. | Borderline, public-sensitive. |
| 88 | `agn_038` | blue jab blocked | Blue jab is visibly extended toward red guard/head. | High-risk real-punch drop, public-sensitive. |
| 150 | `agn_038` | blue uppercut miss | Blue lunges/extends while red ducks; a miss/near miss is plausible. | High-risk public-sensitive drop. |
| 295 | `agn_047` | red cross miss | Far camera, but active exchange and red extension are plausible. | Borderline private-risk drop. |
| 325 | `agn_047` | red cross blocked | Ambiguous frame; not an obvious false positive, but less clear than the others. | Borderline. |
| 326 | `agn_047` | blue cross blocked | Blue extension toward red head/guard is visible. | High-risk real-punch drop. |
| 330 | `agn_047` | red jab miss | Red extension with blue slipping/backing away is plausible. | High-risk real-punch drop. |

Conclusion: top-K token clear is a useful local FP/count signal, but the visual
audit is not clean enough to treat top7/top4 as an automatic upload. The
protected top4 candidate avoids `agn_037/038`, but it still keeps several
visually plausible `agn_047` drops; it should be uploaded only as an explicit
diagnostic/private-risk probe.
