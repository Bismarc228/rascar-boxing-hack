# VideoMAE temporal return - 2026-05-22

Goal: park DINO as a diagnostic sidecar and return to frozen VideoMAE as the
main visual backbone. No Kaggle upload was made.

## Tools

Added fixed-row VideoMAE cache joiners:

```text
tools/extract_videomae_fixed_row_features.py
tools/evaluate_videomae_fixed_row_attribute_model.py
tools/make_videomae_fixed_row_attribute_submission.py
```

They load `.npz + *_index.csv`, align features by `(video_key, id)`, keep the
VideoMAE backbone frozen, and train only a small head. This avoids the previous
mistake of accidentally doing a heavy/full-backbone path.

`tools/extract_videomae_fixed_row_features.py` is the tracked replacement for
the prior branch-only temporal extractor. It uses GPU 1 visibility checks and
defaults to `ffmpeg_cuda` (`hevc_cuvid`) plus CUDA preprocessing.

## OOF Checks

Strict current OOF anchor:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
macro=0.415261
```

Direct pooled cache checks on that anchor regressed:

```text
pooled torch_linear guarded landed->miss/blocked best  0.415055  delta=-0.000206
pooled torch_mlp guarded landed->miss/blocked best     0.414757  delta=-0.000504
```

Temporal-bin cache with flattened `8 x 1536` features and a small GPU MLP head
gave only a micro positive:

```text
temporal flatten torch_mlp effectiveness margin 0.7
guard_from=landed
guard_to=miss,blocked
macro=0.415653
delta=+0.000392
changed=261
```

Saved OOF rows:

```text
data/processed/vit_features/videomae_temporalbins_mlp_eff_landed_to_nonlanded_m07_strict_clearens094012_rows_20260522.csv
```

Verification:

```text
macro_score=0.415653
```

Diff shape:

```text
changes=261
transitions=landed->miss:245,landed->blocked:16
```

Interpretation: the only locally useful VideoMAE temporal signal is the same
narrow landed/non-landed correction seen in dense DINO ROI, but it is much
weaker than DINO ROI and changes many OOF rows.

## Test Artifact

The existing test VideoMAE cache was pooled/direct only:

```text
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_prime_publicbest_test_cuda_preproc_20260522.npz
shape=(739, 1, 1536)
```

The temporal OOF head needs `shape=(N, 8, 1536)`, so a matching test temporal
cache was generated for the current live non-hack anchor. The extractor source
was the prior temporal VideoMAE branch (`7d720e3:tools/extract_vit_candidate_features.py`)
because that source is absent from the current branch. It ran with
`CUDA_VISIBLE_DEVICES=1` and CUDA preprocessing. Note: this legacy extractor is
not the new DINO/RGB `ffmpeg_cuda`/NVDEC path, so future broad VideoMAE
extraction should add a tracked NVDEC VideoMAE extractor rather than relying on
this legacy script.

That follow-up is now implemented as:

```text
tools/extract_videomae_fixed_row_features.py
```

Smoke command:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/extract_videomae_fixed_row_features.py \
  --video-split test \
  --predictions submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --crop-modes attacker_defender,glove_target \
  --clip-len 16 \
  --frame-stride 2 \
  --videomae-pooling temporal \
  --video-decoder ffmpeg_cuda \
  --preprocess-device cuda \
  --device cuda \
  --batch-size 2 \
  --max-rows 2 \
  --output data/processed/vit_features/smoke_videomae_temporal_bins_test2_ffmpegcuda_20260522.npz \
  --index-output data/processed/vit_features/smoke_videomae_temporal_bins_test2_ffmpegcuda_20260522_index.csv
```

Smoke result:

```text
shape=(2, 8, 1536)
cosine_vs_legacy_first2=0.999474,0.999421
mean_abs_diff_vs_legacy_first2=0.000945
```

Full ffmpeg-cuda test cache:

```text
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_temporal_bins_prime_publicbest_blockedonly_test_ffmpegcuda_20260522.npz
shape=(696, 8, 1536)
cosine_vs_legacy_full: min=0.966235 mean=0.998569 max=0.999806
mean_abs_diff_vs_legacy_full=0.001164
```

Test temporal cache:

```text
data/processed/vit_features/videomae_attackerdef_glovetarget_clip16s2_temporal_bins_prime_publicbest_blockedonly_test_cuda_preproc_20260522.npz
shape=(696, 8, 1536)
```

Generated no-upload CSV:

```text
submissions/prime_publicbest_nonhack_blockedonly_videomae_temporalbins_mlp_eff_landed_to_nonlanded_m07_20260522_OFFLINE_CANDIDATE.csv
```

Tracked ffmpeg-cuda regenerated no-upload CSV:

```text
submissions/prime_publicbest_nonhack_blockedonly_videomae_temporalbins_mlp_eff_landed_to_nonlanded_m07_ffmpegcuda_20260522_OFFLINE_CANDIDATE.csv
```

Validation:

```text
Validation passed.
```

Diff versus the live non-hack public anchor:

```text
changed_ids=38
changed_by_column=effectiveness:38
changed_by_video=agn_037:2,agn_038:3,agn_039:3,agn_047:3,agn_048:2,agn_049:1,agn_062:9,agn_063:8,agn_064:7
transitions=landed->miss:27,landed->blocked:11
base_ptype=hook:23,cross:8,jab:5,uppercut:2
base_fighter=blue:23,red:15
```

Diff for the ffmpeg-cuda regenerated CSV:

```text
changed_ids=36
changed_by_column=effectiveness:36
changed_by_video=agn_037:2,agn_038:5,agn_039:2,agn_047:2,agn_048:1,agn_049:1,agn_062:7,agn_063:8,agn_064:8
transitions=landed->miss:25,landed->blocked:11
base_ptype=hook:21,cross:10,jab:3,uppercut:2
changed_vs_legacy_candidate_effectiveness=11
```

## Decision

Do not upload this candidate from current evidence. It is a valid reproducible
VideoMAE temporal artifact, but the OOF gain is only `+0.000392` and the test
changes are still attribute-only on top of the public-sensitive `blocked_only`
anchor. It is weaker than the dense DINO ROI local signal and should be treated
as a conservative reference candidate, not a leaderboard probe.

Next useful VideoMAE work should be one of:

```text
1. restore/add a tracked NVDEC VideoMAE extractor;
2. use the temporal cache for a proper selected-event residual/consistency head;
3. test top-block adapter/LoRA or very small partial fine-tune, still group-safe.
```
