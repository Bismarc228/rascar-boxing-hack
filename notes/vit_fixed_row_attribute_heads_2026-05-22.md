# ViT fixed-row attribute heads, 2026-05-22

Goal: test pretrained ViT/CLIP features as fixed-row component heads for
`punch_type`, `effectiveness`, `hand`, `target`, and `fighter`. This is local
OOF research only. No Kaggle upload was made.

## GPU / CPU diagnosis

The first extractor run was correctly loading CLIP on CUDA, but it was mostly
CPU-bound. The bad path was:

```text
cv2 random frame seek -> CPU crop/resize/cvtColor/normalize -> short GPU CLIP flush
```

The extractor now uses a hybrid sparse reader: wanted frames are grouped into
short sequential runs, while large gaps still use seek. It also enforces GPU-1
visibility for CUDA jobs:

```text
CUDA_VISIBLE_DEVICES=1
extractor_device=cuda:0
device_name=NVIDIA RTX 6000 Ada Generation
```

Standalone CLIP forward on GPU 1 is fast, so the backbone is not the bottleneck:

```text
bs=32  sec=0.347  warmup
bs=64  sec=0.029
bs=128 sec=0.065
bs=256 sec=0.133
```

Full OOF extraction still spends most wall time in CPU decode/crop/resize:

```text
rows=1226
offsets=-4,0,4
crops=2 per offset
flushes=58
GPU flush time approx 0.10-0.18s after warmup
wall time approx 3:19 to 3:21 per crop set
```

Next speed fix should be pipeline overlap or moving resize/normalize to CUDA,
not another backbone change.

Follow-up: `--preprocess-device cuda` moves final resize/normalize from
OpenCV/NumPy to torch/CUDA. It keeps CPU crop/decode and changes resize
interpolation from OpenCV area to torch bilinear, so features are not identical
to the CPU-preprocess cache.

Observed on `attacker_defender,glove_target`, offsets `-4,0,4`:

```text
CPU preprocess wall time   approx 3:21
CUDA preprocess wall time  approx 3:00
```

This is only a moderate speedup because video decode and crop construction are
still CPU-side.

## Commands

Feature cache 1:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/extract_vit_candidate_features.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --backbone clip_vitb16 \
  --crop-modes union,defender_head_body \
  --frame-offsets=-4,0,4 \
  --batch-size 128 \
  --max-sequential-gap 8 \
  --output data/processed/vit_features/clip_vitb16_union_defender_t3_current_oof_hybrid_20260522.npz \
  --index-output data/processed/vit_features/clip_vitb16_union_defender_t3_current_oof_hybrid_20260522_index.csv
```

Feature cache 2:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/extract_vit_candidate_features.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --backbone clip_vitb16 \
  --crop-modes attacker_defender,glove_target \
  --frame-offsets=-4,0,4 \
  --batch-size 128 \
  --max-sequential-gap 8 \
  --output data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_hybrid_20260522.npz \
  --index-output data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_hybrid_20260522_index.csv
```

Best component sweep:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/evaluate_vit_fixed_row_attribute_head.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --feature-cache data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_hybrid_20260522.npz \
  --heads effectiveness,punch_type,hand,target,fighter \
  --margins 0.0,0.2,0.3,0.5,0.7,0.9 \
  --combos 'punch_type:0.2,effectiveness:0.7;punch_type:0.2,effectiveness:0.9;punch_type:0.3,effectiveness:0.7;punch_type:0.2,effectiveness:0.7,target:0.9'
```

CUDA-preprocess sweep:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 .venv/bin/python tools/extract_vit_candidate_features.py \
  --predictions data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_exchange_tracklet_rgb_effectiveness_margin02_oof.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26x_conf035 \
  --backbone clip_vitb16 \
  --crop-modes attacker_defender,glove_target \
  --frame-offsets=-4,0,4 \
  --batch-size 256 \
  --max-sequential-gap 8 \
  --preprocess-device cuda \
  --output data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522.npz \
  --index-output data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_20260522_index.csv
```

## Baseline

Fixed-row OOF baseline:

```text
macro              0.407232
score_time         0.537088
score_fighter      0.544665
score_punch_type   0.203789
score_effectiveness 0.278732
score_hand         0.526763
score_target       0.485253
fp_penalty         0.073644
n_rows             1226
labeled_rows       1051
```

Label counts:

```text
punch_type: cross=378 hook=378 jab=191 uppercut=104
effectiveness: blocked=117 landed=613 miss=321
hand: left=539 right=512
target: body=230 head=821
fighter: blue=575 red=476
```

## Results

`union,defender_head_body` was weak. Only a tiny high-margin effectiveness
move was positive:

```text
effectiveness_margin_0.9 macro=0.407338 delta=+0.000106
score_effectiveness 0.278732 -> 0.280056
changed=7
```

`attacker_defender,glove_target` is the better crop set:

```text
effectiveness_margin_0.7 macro=0.407544 delta=+0.000313
score_effectiveness 0.278732 -> 0.282640
changed=86

punch_type_margin_0.2 macro=0.407835 delta=+0.000603
score_punch_type 0.203789 -> 0.209818
changed=559

combo_punch_type0.2_effectiveness0.7 macro=0.408147 delta=+0.000916
score_punch_type 0.203789 -> 0.209818
score_effectiveness 0.278732 -> 0.282640
changed=645

combo_punch_type0.2_effectiveness0.7_target0.9 macro=0.408157 delta=+0.000926
score_target 0.485253 -> 0.485419
changed=674
```

The `target:0.9` increment is only `+0.000010` macro and should be treated as
noise until a second split or model confirms it.

CUDA-preprocess changed the best point. It improved the `punch_type` signal but
lost the positive `effectiveness` gate:

```text
punch_type_margin_0.2 macro=0.408399 delta=+0.001168
score_punch_type 0.203789 -> 0.215465
changed=537
wins=8

effectiveness_margin_0.7 macro=0.407115 delta=-0.000117
score_effectiveness 0.278732 -> 0.277272
changed=71

combo_punch_type0.2_effectiveness0.7 macro=0.408283 delta=+0.001051
score_punch_type 0.203789 -> 0.215465
score_effectiveness 0.278732 -> 0.277272
changed=608
```

Use the CUDA-preprocess `punch_type:0.2` point as the current best CLIP fixed-row
component result. Do not add the `effectiveness` head from this cache.

Hand, target, and fighter direct CLIP heads are not useful yet:

```text
hand_margin_0.7    macro=0.406575 delta=-0.000656
target_margin_0.7  macro=0.406197 delta=-0.001035
fighter_margin_0.9 macro=0.403424 delta=-0.003807
```

## Matched-row diagnostics

For `attacker_defender,glove_target`, raw model diagnostics on matched rows:

```text
effectiveness baseline: accuracy=0.544244 balanced_acc=0.459302 macro_f1=0.445687 weighted_f1=0.545601
effectiveness raw:      accuracy=0.490961 balanced_acc=0.457857 macro_f1=0.435370 weighted_f1=0.505654
  blocked f1=0.297735 recall=0.393162 support=117
  landed  f1=0.592189 recall=0.531811 support=613
  miss    f1=0.416185 recall=0.448598 support=321

punch_type baseline: accuracy=0.397716 balanced_acc=0.310421 macro_f1=0.305457 weighted_f1=0.383277
punch_type raw:      accuracy=0.341579 balanced_acc=0.342558 macro_f1=0.320268 weighted_f1=0.346459
  cross    f1=0.326797 recall=0.264550 support=378
  hook     f1=0.410256 recall=0.380952 support=378
  jab      f1=0.332696 recall=0.455497 support=191
  uppercut f1=0.211321 recall=0.269231 support=104
```

Interpretation: raw CLIP hurts majority-class accuracy, but it has useful
balanced signal for `punch_type`, especially `jab`/`uppercut` recall. The
margin gate is carrying the positive macro result.

CUDA-preprocess `punch_type` raw diagnostics:

```text
punch_type baseline: accuracy=0.397716 balanced_acc=0.310421 macro_f1=0.305457 weighted_f1=0.383277
punch_type raw:      accuracy=0.358706 balanced_acc=0.352967 macro_f1=0.334010 weighted_f1=0.361004
  cross    f1=0.328308 recall=0.259259 support=378
  hook     f1=0.448000 recall=0.444444 support=378
  jab      f1=0.319066 recall=0.429319 support=191
  uppercut f1=0.240664 recall=0.278846 support=104
```

Weak head diagnostics:

```text
hand baseline macro_f1=0.860948, raw=0.451758
target baseline macro_f1=0.633277, raw=0.521205
fighter baseline macro_f1=0.881423, raw=0.651842
```

These should not be global CLIP replacement heads. For fighter, use per-video
identity/tracklet calibration. For hand and target, use pose trajectory plus
very targeted visual witnesses.

## Conclusions

- The user concern was valid: the initial run looked CPU-only because the input
  pipeline starved CUDA. CLIP forward is on GPU 1 and fast; decode/preprocess
  remains CPU-bound.
- Pretrained CLIP crop features are a real but small fixed-row attribute signal
  for `punch_type`.
- The best current component result is CUDA-preprocess `punch_type:0.2`, macro
  `0.408399`, delta `+0.001168` on this OOF source.
- CPU-preprocess also had a small `effectiveness` result, but the signal did not
  survive the CUDA-preprocess variant; treat `effectiveness` as unresolved until
  VideoMAE or a second split confirms it.
- This is below a submit-level gate and is not a Kaggle candidate by itself.
- Next model step: VideoMAE or sparse ViT sequence features for
  `punch_type/effectiveness`, plus pose/audio features in the same calibrated
  head. Next systems step: overlap CPU loader and GPU inference or move
  resize/normalize to CUDA.
