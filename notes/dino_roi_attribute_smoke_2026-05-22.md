# DINO ROI Attribute Smoke, 2026-05-22

Goal: re-test DINO in a formulation closer to what the benchmark papers imply:
frozen dense patch tokens, local ROI pooling around boxing pose geometry, and a
small head only. No Kaggle upload was made from this experiment.

## External Check

Primary sources support treating DINO as a dense frozen backbone, not just as a
single global crop descriptor:

```text
https://arxiv.org/abs/2304.07193
https://arxiv.org/abs/2508.10104
```

DINOv2 reports strong image-level and pixel-level all-purpose features, and
DINOv3 emphasizes high-quality dense features. For this task, the efficient
variant is therefore DINO patch-token ROI pooling around the attacker wrist and
opponent head/body, with DINO frozen and only the classifier head trained.

## NVIDIA Decode Rule

The new tools use the same NVIDIA decode path as the updated RGB tools:

```text
ffmpeg -hwaccel cuda -c:v hevc_cuvid ... -pix_fmt nv12
```

All commands below used:

```text
CUDA_VISIBLE_DEVICES=1
```

No OpenCV CPU video decode was used.

## Implementation

Added:

```text
tools/evaluate_dino_roi_attribute_model.py
tools/make_dino_roi_attribute_submission.py
```

The extractor:

- decodes frames with `hevc_cuvid` into NV12;
- converts NV12 to RGB on GPU;
- crops the union fighter box and resizes to `518`;
- runs frozen `vit_small_patch14_dinov2.lvd142m`;
- reads `forward_features` patch tokens;
- pools local tokens at attacker wrist, opponent head, opponent body, selected
  target, and wrist-target midpoint;
- trains only a `torch_linear` head with GPU PCA bottleneck.

## OOF Command

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/evaluate_dino_roi_attribute_model.py \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --feature-cache data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_oof_20260522.npz \
  --model-name vit_small_patch14_dinov2.lvd142m \
  --image-size 518 \
  --clip-len 8 \
  --frame-stride 2 \
  --crop-expand 0.22 \
  --roi-radius 1 \
  --device cuda \
  --batch-size 32 \
  --cpu-threads 1 \
  --train-columns punch_type,effectiveness,target \
  --margin-columns punch_type,effectiveness,target \
  --head-type torch_linear \
  --torch-pca-components 128 \
  --torch-epochs 80 \
  --torch-batch-size 256 \
  --torch-lr 0.002 \
  --torch-weight-decay 0.001
```

Feature cache:

```text
data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_oof_20260522.npz
features=(1188, 8, 4233)
```

Extraction took about 2.2 minutes for the 13 OOF train videos.

## OOF Result

Baseline strict clearens094/012 source:

```text
macro=0.415261
type=0.211716
effectiveness=0.289848
hand=0.521370
target=0.478143
```

Best DINO ROI result:

```text
effectiveness_margin_0.5
macro=0.416799
delta=+0.001538
effectiveness=0.309074
changed=238
```

Other useful signals:

```text
effectiveness_margin_0.3   macro=0.416563  delta=+0.001302
punch_type_margin_0.3      macro=0.415941  delta=+0.000680
punch_type_margin_0.2      macro=0.415821  delta=+0.000560
effectiveness_margin_0.4   macro=0.416480  delta=+0.001219
effectiveness_margin_0.6   macro=0.416720  delta=+0.001458
```

Target replacement is still harmful:

```text
target_margin_0.4          macro=0.413700  delta=-0.001561
hand_target                macro=0.411188  delta=-0.004073
```

Saved OOF rows:

```text
data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_eff_margin03_rows_20260522.csv
data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_eff_margin05_rows_20260522.csv
```

Protected controls show the lift depends on changing rows currently labeled
`landed`:

```text
protect_landed             macro=0.414819  delta=-0.000443
miss_only                  macro=0.415167  delta=-0.000094
blocked_only               macro=0.414913  delta=-0.000349
non_landed_to_landed_only  macro=0.414790  delta=-0.000471
```

For the stronger margin `0.5`, the effect is clearer:

```text
full_m05                  macro=0.416799  delta=+0.001538  changed=238
landed_to_nonlanded      macro=0.416784  delta=+0.001523  changed=117
landed_to_miss           macro=0.416159  delta=+0.000898  changed=83
landed_to_blocked        macro=0.415886  delta=+0.000624  changed=34
nonlanded_changes        macro=0.415276  delta=+0.000015  changed=121
```

The useful local signal is almost entirely `landed -> miss/blocked`. This
makes the full test artifact too broad, but gives a narrower no-upload artifact
to audit.

## Test Artifact

Prepared on top of the current live public-best CSV, but not uploaded:

```text
submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m03_20260522_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m05_20260522_OFFLINE_CANDIDATE.csv
submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m05_landed_to_nonlanded_20260522_OFFLINE_CANDIDATE.csv
```

Generation command:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/make_dino_roi_attribute_submission.py \
  --train-predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --train-tracks-dir data/processed/pose_tracks/train_yolo11s_conf035 \
  --train-feature-cache data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_strict_clearens094012_oof_20260522.npz \
  --input submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_abl_blocked_only_20260522_OFFLINE_CANDIDATE.csv \
  --test-tracks-dir data/processed/pose_tracks/test_yolo11s_conf035 \
  --test-feature-cache data/processed/vit_features/dinov2_vits14_roi_clip8_stride2_blockedonly_test_20260522.npz \
  --output submissions/prime_publicbest_nonhack_clearens094012_blockedonly_dinov2roi_eff_m03_20260522_OFFLINE_CANDIDATE.csv \
  --column effectiveness \
  --margin 0.3 \
  --model-name vit_small_patch14_dinov2.lvd142m \
  --image-size 518 \
  --clip-len 8 \
  --frame-stride 2 \
  --crop-expand 0.22 \
  --roi-radius 1 \
  --device cuda \
  --batch-size 32 \
  --cpu-threads 1 \
  --head-type torch_linear \
  --torch-pca-components 128 \
  --torch-epochs 80 \
  --torch-batch-size 256 \
  --torch-lr 0.002 \
  --torch-weight-decay 0.001
```

Validation:

```text
Validation passed.
```

Diff for broad margin `0.3` against the current live `blocked_only` CSV:

```text
changed_ids=213
changed_by_column=effectiveness:213
changed_by_video=agn_047:43,agn_063:40,agn_038:23,agn_062:23,agn_049:21,agn_048:20,agn_064:16,agn_039:15,agn_037:12
changed_by_base_effectiveness=miss:75,blocked:70,landed:68
top_changes=miss->landed:53,landed->miss:52,blocked->landed:39,blocked->miss:31,miss->blocked:22,landed->blocked:16
```

Diff for broad margin `0.5`:

```text
changed_ids=134
changed_by_column=effectiveness:134
changed_by_video=agn_047:30,agn_063:30,agn_038:14,agn_048:14,agn_049:12,agn_062:11,agn_064:9,agn_037:7,agn_039:7
changed_by_base_effectiveness=miss:52,blocked:45,landed:37
top_changes=miss->landed:37,landed->miss:29,blocked->landed:26,blocked->miss:19,miss->blocked:15,landed->blocked:8
```

Diff for the narrow margin `0.5` landed-to-nonlanded artifact:

```text
changed_ids=37
changed_by_video=agn_063:10,agn_047:7,agn_062:5,agn_064:5,agn_048:4,agn_037:2,agn_038:2,agn_039:1,agn_049:1
top_changes=landed->miss:29,landed->blocked:8
agn_037_038_changes=4
```

## Decision

The user's suspicion was correct: the earlier DINO tests were underusing the
model. Dense ROI DINOv2 is the first DINO formulation in this branch that gives
a meaningful OOF gain on the strong strict clearens source.

Do not use this as an automatic upload trigger. The best local signal now points
specifically to `landed -> miss/blocked` corrections, and the narrow test CSV
changes only 37 rows, but this is still an attribute-only private-risk artifact
stacked on a public-sensitive clear/drop anchor. Treat it as a strong research
lead for ROI effectiveness/contact modeling, not as a safe public/private anchor
yet.
