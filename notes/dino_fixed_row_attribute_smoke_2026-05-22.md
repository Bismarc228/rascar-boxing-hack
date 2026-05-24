# DINO fixed-row attribute smoke - 2026-05-22

This is a no-upload experiment for replacing or complementing the current
VideoMAE fixed-row attribute path with DINO-family crop features. All GPU work
used `CUDA_VISIBLE_DEVICES=1`. No Kaggle submission was made.

Important implementation detail: DINO/DINOv2 was not fine-tuned. The backbone
was used only once to extract frozen crop features into `.npz` caches. The
attribute experiments below train heads on top of those frozen features.

## Environment

The system Python currently has an incompatible `transformers` /
`huggingface-hub` pair, but the project `.venv` is usable:

```text
torch=2.6.0+cu124
torchvision=0.21.0+cu124
transformers=4.57.6
huggingface_hub=0.36.2
timm=1.0.27
```

DINOv1 `vit_small_patch16_224.dino` required falling back from
`global_pool=avg` to `global_pool=token` because the pretrained checkpoint has
`fc_norm`/`norm` key differences. DINOv2
`vit_small_patch14_dinov2.lvd142m` requires `--image-size 518`; `224` fails a
model input assertion.

## Feature Caches

Added `tools/extract_rgb_candidate_features.py` to reconstruct the same
pose-candidate pool used by `tools/make_rgb_fixed_row_attribute_submission.py`
and write a reusable RGB feature cache plus index CSV.

Extended `tools/evaluate_rgb_fixed_row_attribute_model.py` so frozen-cache heads
can run either through the old sklearn `StandardScaler+PCA+LogReg` path or a
GPU `torch_linear` / residual `torch_mlp` path with optional torch PCA.
Extended the submission/extraction path with `--gpu-preprocess` and
`--video-decoder ffmpeg_cuda`. The latter uses `hevc_cuvid` and pipes NV12
frames; YUV-to-RGB conversion, crop, resize, normalization, DINO, and the torch
head run on GPU. This avoids the earlier OpenCV/BGR path that used several CPU
cores for decode/color/resize. CPU is still used for process control, bbox
lookup, and the raw frame pipe.

Generated caches:

```text
data/processed/vit_features/dino_vits16_union_clip4_stride2_pool1800_current_oof_20260522.npz
data/processed/vit_features/dino_vits16_union_clip4_stride2_pool1800_current_oof_20260522_index.csv

data/processed/vit_features/dinov2_vits14_union_clip4_stride2_pool1800_current_oof_20260522.npz
data/processed/vit_features/dinov2_vits14_union_clip4_stride2_pool1800_current_oof_20260522_index.csv
data/processed/vit_features/dinov2_vits14_union_clip4_stride2_pool1800_prime_publicbest_test_20260522.npz
```

Both caches have `features=(22691, 4, 384)`. DINOv2 extraction at 518px took
about 17.5 minutes on the allowed GPU for the original OOF OpenCV path. The
test-side DINOv2 cache with `ffmpeg_cuda`/NV12 GPU preprocessing took about
3.5 minutes for 9 test videos.

## Direct Replacement Results

On the older `audio_tracklet` baseline (`0.405200`), DINOv2 is useful but not
decisive:

```text
DINOv1 effectiveness margin 0.2       0.406806  delta=+0.001606
DINOv2 effectiveness margin 0.2       0.407452  delta=+0.002251
DINOv2 ptype_eff                      0.408302  delta=+0.003102
```

Saved OOF rows:

```text
data/processed/vit_features/dino_vits16_union_clip4_stride2_pool1800_current_oof_effectiveness_margin02_rows_20260522.csv
data/processed/vit_features/dinov2_vits14_union_clip4_stride2_pool1800_current_oof_pca96_c015_effectiveness_margin02_rows_20260522.csv
data/processed/vit_features/dinov2_vits14_union_clip4_stride2_pool1800_current_oof_pca96_c015_ptype_eff_rows_20260522.csv
```

Directly swapping DINOv2 into the stronger `clip_ptype_m028` path does not
replace VideoMAE effectiveness:

```text
clip_ptype_m028 baseline              0.408457
DINOv2 effectiveness margin 0.3       0.408877  delta=+0.000420
VideoMAE direct effectiveness         0.409796
VideoMAE d2res effectiveness          0.411357
```

DINOv2 has a positive `punch_type` signal on that same baseline:

```text
DINOv2 punch_type                     0.409516  delta=+0.001059
```

Saved replacement row:

```text
data/processed/vit_features/clip_ptype_m028_dinov2_vits14_pca96_c015_eff_margin030_rows_20260522.csv
```

## Complement Results

Stacking DINOv2 over the current VideoMAE-style rows confirms the split:
`effectiveness` mostly regresses, while `punch_type` can help.

```text
VideoMAE d2res baseline               0.411357
DINOv2 punch_type                     0.411533  delta=+0.000176
DINOv2 effectiveness margin 0.3       0.410949  delta=-0.000408
DINOv2 ptype_eff                      0.409011  delta=-0.002346

clearens09 baseline                   0.414485
DINOv2 punch_type                     0.415588  delta=+0.001103
DINOv2 effectiveness margin 0.3       0.413905  delta=-0.000580
DINOv2 ptype_eff                      0.413649  delta=-0.000836

strict clearens094/012 baseline       0.415261
DINOv2 punch_type                     0.415445  delta=+0.000184
DINOv2 effectiveness margin 0.3       0.415450  delta=+0.000189
DINOv2 ptype_eff                      0.413287  delta=-0.001974
```

Best saved DINO complement row:

```text
data/processed/vit_features/clip_ptype_m028_videomae_clearens09_dinov2_vits14_pca96_c015_ptype_rows_20260522.csv
```

It verifies at:

```text
macro_score=0.415588
```

## GPU Head-Only Check

The positive sklearn result above was still head-only, but sklearn runs on CPU.
After adding the torch head path, the same frozen DINOv2 cache was tested with
GPU heads (`torch_head_device=cuda`, physical GPU 1 through
`CUDA_VISIBLE_DEVICES=1`).

Raw high-dimensional GPU heads overfit/noise-amplify the DINO features:

```text
torch_mlp h192 d2 e80 punch_type       0.410221  delta=-0.004264
torch_linear e300 punch_type           0.411282  delta=-0.003203
torch_linear e80 ptype margin 0.4      0.413123  delta=-0.001362
```

Adding a torch PCA bottleneck before the GPU linear head makes the result
positive, but still smaller than sklearn:

```text
torch_linear pca96 e250 punch_type direct      0.413741  delta=-0.000744
torch_linear pca96 e250 ptype margin 0.4       0.414679  delta=+0.000194
torch_mlp pca96 h128 d1 e120 ptype margin 0.4  0.411196  delta=-0.003289
```

Saved GPU-head row:

```text
data/processed/vit_features/clip_ptype_m028_videomae_clearens09_dinov2_vits14_torchlinear_pca96_e250_ptype_margin04_rows_20260522.csv
```

It verifies at:

```text
macro_score=0.414679
```

The same GPU-head recipe on the stricter live clearens094/012 OOF source is
also only a micro-signal:

```text
strict clearens094/012 baseline       0.415261
torch_linear pca96 e250 ptype margin 0.4       0.415539  delta=+0.000278
```

## Test-Side No-Upload Candidate

Generated and validated a matching test-side artifact on top of the current
live non-hack public anchor:

```text
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_dinov2_ptype_torchlinear_pca96_m04_20260522_OFFLINE_CANDIDATE.csv
```

Validation:

```text
Validation passed.
```

Diff versus the strict clearens094/012 anchor:

```text
changed_ids=165
changed_by_column=punch_type:165
changed_by_video=agn_037:16,agn_038:14,agn_039:14,agn_047:30,agn_048:9,agn_049:17,agn_062:26,agn_063:26,agn_064:13
changed_by_base_effectiveness=blocked:35,landed:77,miss:53
top_ptype_changes=hook->cross:30,hook->uppercut:23,jab->cross:19,hook->jab:19,cross->jab:19,cross->hook:19
```

This is not upload-recommended from the current evidence. The OOF lift on the
strict anchor is only `+0.000278`, and the test artifact rewrites 165
`punch_type` values including 77 rows currently marked `landed`.

## Decision

Do not replace the VideoMAE effectiveness head with DINOv2. The direct
DINOv2-effectiveness replacement is far behind the VideoMAE d2res head on OOF.

DINOv2 is still worth keeping as a `punch_type` feature source, but the current
GPU torch heads are weaker than the CPU sklearn PCA+LogReg head. The generated
test artifact is valid and reproducible but private-risk, not an upload trigger.
Timing is still not addressed by this fixed-row attribute experiment; for
timing, the next model should be sequence-native (ResNet/LSTM/Transformer-style)
rather than another per-row crop classifier.
