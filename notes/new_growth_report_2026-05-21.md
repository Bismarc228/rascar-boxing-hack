# New Growth Report - 2026-05-21

## Executive Summary

Текущий public best: `0.16461` на ветке решений
`hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088`. Проверено через
Kaggle CLI 2026-05-21: команда `Los Pollos Hermanos` лидирует с `0.16461`,
следующий видимый score `0.04481`; использовано `8/30` сабмитов за UTC-день,
но рабочий пользовательский лимит остается строже: максимум 2 новых сабмита
только под явный прорыв.

Главный вывод: очередные локальные threshold/NMS/sequence sweeps почти точно
не дадут надежный public рост. Лучшие новые направления должны ломать один из
текущих потолков:

1. **Проверить, действительно ли `sample_submission` ограничивает число строк
   на видео.** Сейчас все генераторы зажаты per-video capacity из sample.
   Если Kaggle metric использует `video_key` из строки и не привязывает
   `id -> video_key`, можно перераспределить неиспользованные строки между
   видео. Это high-risk/high-upside гипотеза, независимая от pose-моделей.
2. **Сделать fight-level, а не video-level логику.** Test состоит из трех
   полных боев по 3 раунда. Текущий pipeline принимает решения почти по
   отдельным видео. Нужно использовать общие identity/count/style признаки
   внутри боя.
3. **Добавить настоящий RGB/video event spotter как независимый witness.**
   Текущий лучший learned path - pose-sequence TCN. RGB-клип вокруг обмена,
   контакта и защиты еще не проверен как основной новый сигнал.
4. **Исправлять fighter identity только через tracklet/appearance по всему
   бою.** Локальный fighter oracle дает headroom, но текущие cached-color,
   bbox-clustering и ResNet50-crop подходы уже регресснули.

Не стоит тратить время/сабмиты на full sequence replacement, yolo26x agreement,
dense recall, global swaps/offsets, audio-only, attribute-only, current flat
rankers и повторные public probes.

## Evidence Snapshot

- Competition task: 9 test clips, 1594 fixed rows, metric weights:
  `time 50%`, `fighter 20%`, `punch_type 10%`, `effectiveness 8%`,
  `hand/target 6%` each, minus FP penalty.
- `clear=false` rows are safe filler: they do not match and do not count as FP.
- Current public results after reset:
  - `0.16461`: yolo26l public base + sequence only on `agn_038`.
  - `0.12712`: full sequence snap4.
  - `0.10784`: full yolo26x+yolo11s agreement.
  - `0.12888`: yolo26x+yolo11s agreement only on `agn_038`.
  - adding sequence on `agn_047`, `agn_062`, `agn_063` stayed `0.16461`.
  - adding sequence on `agn_037` dropped to `0.12712`.
- Validation rows:
  - yolo26l public-base OOF: `0.365177`.
  - full sequence snap4 OOF: `0.379682`, better timing but higher FP.
  - gated yolo26l/sequence OOF: `0.389125`; swept gate reported `0.395013`.
- Test row counts currently selected:

| video | root | base | current best | gated candidate | full sequence |
| --- | --- | ---: | ---: | ---: | ---: |
| `agn_037` | Турнир Бокс | 52 | 52 | 52 | 52 |
| `agn_038` | Турнир Бокс | 74 | 81 | 81 | 81 |
| `agn_039` | Турнир Бокс | 57 | 57 | 57 | 57 |
| `agn_047` | Турнир Бокс 2 | 112 | 112 | 112 | 125 |
| `agn_048` | Турнир Бокс 2 | 44 | 44 | 44 | 44 |
| `agn_049` | Турнир Бокс 2 | 62 | 62 | 62 | 62 |
| `agn_062` | Турнир Бокс 2 | 113 | 113 | 125 | 125 |
| `agn_063` | Турнир Бокс 2 | 107 | 107 | 123 | 123 |
| `agn_064` | Турнир Бокс 2 | 75 | 75 | 75 | 75 |

Important: these counts are clipped by per-video sample capacity. For example,
`agn_037` has only 52 sample rows although its root-rate expectation from train
duration is about 131 clear punches. This may be a real cap, or it may be an
unnecessary self-imposed cap from local validation/generators.

## Highest-Upside New Hypotheses

### 1. Relaxed Row/ID Capacity Reallocation

**Hypothesis.** Kaggle requires the same row count and `id` set, but may not
require each `id` to keep the original sample `video_key`. The task text says
row order does not affect evaluation and includes `video_key` as a submission
column. Local `validate_submission(..., strict_id_metadata=True)` is stricter
than that interpretation.

**Why it can grow.**

- Current generators cap each video at the number of sample rows for that video.
- Several likely-important videos are capped hard: `agn_037=52`, `agn_048=44`,
  `agn_049=62`, `agn_064=75`.
- `agn_062` has 746 sample rows, leaving hundreds of unused `clear=false`
  filler rows that could theoretically be reallocated.
- If hidden GT counts are closer to root/fight rates than sample capacity, the
  current system is structurally unable to recover recall on capped videos.

**Why it can fail badly.**

- Kaggle evaluator may enforce `id -> video_key` implicitly or reject/score rows
  differently from local assumptions.
- Even if accepted, reallocating rows can increase FP fast. Dense recall already
  failed publicly.
- It may damage private if `agn_062` really has many hidden events and spare
  rows are not actually spare in the optimal solution.

**Recommended next step.**

Build a separate `relaxed_metadata` submission generator that keeps the same
1594 `id`s but allows video reassignment, then validate with
`strict_id_metadata=False`. Do not upload without explicit approval. The first
safe experiment should move only low-confidence unused rows into a conservative
set of extra predictions for `agn_037` or another capped video, preserving the
known-good `agn_038` replacement.

### 2. Fight-Level Policy Across Rounds

**Hypothesis.** The three rounds of the same fight share fighter appearance,
pace, stance, referee/camera conditions, and punch-rate regime. Current logic is
mostly per-video and loses that structure.

**Why it can grow.**

- Public evidence is fight-local: `agn_038` sequence is good, `agn_037`
  sequence is bad, both are in the same fight.
- Validation gate mistakes such as `agn_056` show nearest-frame/count features
  are insufficient; the decision needs richer fight/style context.
- Test contains exactly three complete fights: `37-39`, `47-49`, `62-64`.

**Concrete version.**

- Build per-fight features: exchange density, mean candidate score, pose
  coverage, role switch rate, boxer distance distribution, count per active
  minute, base-vs-sequence agreement tails, and round-to-round count ratios.
- Train/evaluate leave-one-fight OOF policy choosing base/sequence/cropmotion
  source per video or per event cluster.
- Force known constraints: preserve `agn_038` sequence, reject sequence on
  `agn_037` unless a new signal explains why it is fixed.

**Expected upside.** Medium to high. It may turn the current private-risk gate
(`agn_062/063`) into a more defensible private candidate and may prevent
sequence failures like `agn_056`.

**Bad version.** Another Ridge policy on the current tiny feature table. That
was already worse than threshold gate. The new value must come from new
fight-level features, not a different regressor.

### 3. RGB/Video Event Spotter

**Hypothesis.** Pose-only misses contact, guard/feint distinction, glove impact,
referee occlusion, and blocking. A small RGB/video spotter can be an independent
timing and confidence witness.

**Why it can grow.**

- Time is 50% of the metric; current gated validation still has average timing
  score around `0.51`, leaving large room.
- Existing crop-motion showed weak independent signal but was too crude.
- The `.venv` already has `torch`, `torchvision`, `timm`, `decord`, `av`,
  `transformers`, `librosa`, and `opencv`, so the environment can support a
  frozen-embedding or light fine-tune smoke.
- External work supports the direction:
  - E2E-Spot targets temporally precise fine-grained event spotting in video:
    https://arxiv.org/abs/2207.10213
  - VideoMAE is designed for data-efficient video pretraining and transfer:
    https://arxiv.org/abs/2203.12602
  - T-DEED is a sports precise-event spotting implementation:
    https://github.com/arturxe2/t-deed

**Concrete version.**

- Start with frozen RGB clip embeddings around existing candidate clusters:
  full-ring crop plus attacker/opponent union crop, `t-12..t+12`, 8-16 frames.
- Train only a light head OOF by fight: event probability, offset to impact,
  and maybe confidence calibration.
- Use it as a witness/gate on top of yolo26l/sequence candidates, not as a
  full dense sliding detector on day one.

**Expected upside.** High if the model learns guard/contact/impact cues; medium
if it only recovers the same motion as pose.

**Bad version.** Fine-tuning a large video transformer directly on all frames
with weak OOF. The dataset is small, public already punishes overfit, and the
compute cost is high.

### 4. Tracklet-Level Fighter Identity Across The Whole Fight

**Hypothesis.** Fighter identity headroom is real, but current features are too
weak. Need persistent tracklet/appearance identity across rounds, not per-frame
HSV or simple crop clustering.

**Why it can grow.**

- Event-level matched-fighter oracle gives about `+0.018` macro on strong
  anchors.
- Whole-video swaps, cached HGB/logreg flip models, bbox ROI clustering, and
  ResNet50 crop clustering failed. That means the old feature family is dead,
  not the problem.
- A video segmentation/tracking foundation model can provide cleaner fighter
  masks and stable tracklets. SAM 2 is explicitly designed for promptable image
  and video segmentation/tracking: https://ai.meta.com/sam2/

**Concrete version.**

- Prompt/initialize from high-confidence YOLO fighter bboxes, propagate masks
  through each round, and aggregate per-tracklet appearance from shorts/torso/
  glove regions.
- Learn/test only fighter label changes with timing/count fixed.
- Report `score_fighter`, macro, confusion matrix, and root/video deltas.

**Expected upside.** Medium. Fighter has 20% weight, but label changes are risky
and need high precision.

**Bad version.** More global HSV thresholds or per-video keep/swap. Already
killed.

**2026-05-21 audit update.** A fixed-row fighter audit confirms the identity
headroom but kills simple role/color correction again. On the gated
yolo26l/sequence hybrid, matched-fighter oracle improves `0.389125 -> 0.402894`,
while track role/color/frame-color flips regress or stay neutral. A narrow
opposite-fighter rival-candidate rule is weakly positive across row sources
(`+0.00027` yolo26l, `+0.00044` sequence, `+0.00055` gated hybrid) and is saved
as an ensemble micro-signal, not a submit branch. The next real identity branch
should classify exchange side or build persistent mask/tracklet identity; do
not spend more cycles on global equipment-color flips without new features.

### 5. Exchange-State / No-Punch Negative Gate

**Hypothesis.** The model needs to know when no real exchange is happening:
guards, footwork, referee interruptions, clinch, and recovery create pose
peaks. Current NMS/count controls are indirect.

**Concrete version.**

- Build per-frame exchange state from boxer distance, both-wrist activity,
  mutual forward motion, referee overlap, and ring-position stability.
- Use it as a multiplicative gate on candidate confidence and as count
  allocator per active segment.
- Evaluate fixed-count first to isolate quality from recall.

**Expected upside.** Medium. It can reduce FP without forcing global thresholds.

**Bad version.** Full-frame motion-only detector. Prior motion/audio-only
families were noisy and low precision.

**2026-05-21 audit update.** A fixed-row exchange/no-punch HGB gate is positive
as a validation source: yolo26l base `0.365177 -> 0.370306`, sequence
`0.379682 -> 0.383814`, and hybrid+rival `0.389672 -> 0.390962`. The full
source oracle with base/seq/motion/hybrid-rival/exchange reaches `0.403225`.
The plain gated hybrid does not improve, and the train-all test artifact at the
OOF-selected threshold drops only one row, so this is a source-gating feature
rather than a direct upload.

### 6. Attribute Clip Classifier After Timing Stabilizes

**Hypothesis.** Attributes are not the main bottleneck now, but `punch_type` and
`effectiveness` use balanced components, so rare `uppercut`/`blocked` can matter
once timing/fighter improves.

**Expected upside.** Low to medium now; higher after event selection improves.

**Good version.** Train on matched high-quality clips only; use RGB/pose
features around selected events; report gains with fixed timing.

**Bad version.** Attribute-only submit or nearest-row attribute transfer. Both
are too small by existing evidence.

### 7. Human/Manual Test Annotation

**Hypothesis.** There are only 9 test videos. If the competition rules allow
manual inspection/annotation of test videos, human-aided correction of event
frames and fighter labels could beat all algorithmic tweaks.

**Caution.** This may be disallowed by competition-specific rules. Some Kaggle
rules explicitly ban hand labeling or human prediction of test records, while
others do not. Do not use this route until the exact rules for this competition
are checked and accepted.

**Expected upside.** Very high if allowed; zero if disallowed.

## What Will Likely Work Badly

- Full sequence-TCN replacement: public `0.12712`; keep only as video-local
  source.
- Full yolo26x+yolo11s agreement: public `0.10784`; even `agn_038` hybrid was
  only `0.12888`.
- Dense recall / threshold-count candidates: public punished them heavily.
- More yolo26l alpha/window/root-rate micro-sweeps: prior variants clustered or
  regressed; not enough for the two-submit cap.
- Global frame offsets: tested and worse/neutral.
- Global fighter swaps or per-video keep/swap: tested and bad.
- Current cached-feature fighter flip models: oracle has headroom, models did
  not recover it.
- Simple bbox color clustering and ImageNet ResNet50 crop clustering: both
  regressed.
- Audio-only, hard audio snapping, direct audio rescore: tested negative.
- Motion-only submissions: too noisy; crop-motion is only a weak ensemble
  feature.
- Attribute-only models or nearest attribute transfer: too little metric impact
  while timing is still weak.
- Public probing under the current cap without a concrete local reason.

## Recommended Next Work Order

1. **No-submit capacity audit.**
   - Add/clone a generator that can reassign spare sample ids across videos.
   - Keep official `id` set and row count.
   - Validate with relaxed local metadata check.
   - Produce one conservative diagnostic CSV, but do not upload without explicit
     approval.

2. **Fight-level feature table.**
   - Build per-video/per-fight diagnostics for base, sequence, cropmotion and
     maybe yolo26x witness.
   - Use leave-one-fight validation, not video-only CV.
   - Try to explain `agn_056` and `agn_037` failures before touching submit.

3. **RGB clip witness smoke.**
   - Use `.venv/bin/python`, GPU 1 only if GPU is needed.
   - Freeze a pretrained video/RGB encoder first.
   - Train a light OOF head for candidate probability and offset.
   - Submit only if it improves validation with fixed or lower FP and wins most
     tournament videos.

4. **Tracklet identity prototype.**
   - Use fixed selected events.
   - Extract higher-quality tracklet/mask appearance over whole fights.
   - Only allow high-confidence event-level fighter changes.

5. **Private-risk submit decision.**
   - Existing gated candidate
     `hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv`
     is plausible for private score, but likely public-neutral. Use only if the
   user explicitly accepts private-risk and no stronger new signal appears.

## Execution Update

Implemented the first two recommended no-submit steps on this branch.

### Relaxed Capacity Diagnostic

Created a diagnostic generator:

```text
tools/make_relaxed_capacity_submission.py
```

Generated artifact:

```text
submissions/relaxed_capacity_agn037_plus20_from_agn062_yolo26l_OFFLINE_DIAGNOSTIC.csv
```

What it does:

- Keeps the official `1594` rows and exact `id` set.
- Reassigns `20` unused `clear=false` filler ids from `agn_062` to extra
  conservative yolo26l predictions for capped `agn_037`.
- Preserves the current best `agn_038` sequence replacement.
- Changes clear counts from `agn_037:52` to `agn_037:72`; total clear rows
  become `723`.

Validation result:

```text
python3 tools/validate_data.py --data-root data/raw \
  --submission submissions/relaxed_capacity_agn037_plus20_from_agn062_yolo26l_OFFLINE_DIAGNOSTIC.csv \
  --no-strict-id-metadata
# Validation passed.

python3 tools/validate_data.py --data-root data/raw \
  --submission submissions/relaxed_capacity_agn037_plus20_from_agn062_yolo26l_OFFLINE_DIAGNOSTIC.csv
# Fails strict id metadata, starting at id=678 expected agn_062 but got agn_037.
```

This confirms the relaxed-format hypothesis is technically representable
locally, but it remains high-risk and should not be uploaded without explicit
approval because Kaggle may enforce `id -> video_key` metadata implicitly.

### Fight-Level Source Diagnostics

Created:

```text
tools/build_fight_level_source_table.py
```

Generated:

```text
data/processed/diagnostics/fight_level_sources_validation.csv
data/processed/diagnostics/fight_level_sources_test.csv
```

Validation source summary:

```text
base    0.365177
seq     0.379682  wins 9/13 vs base
motion  0.368244  wins 7/13 vs base
```

Per-video source oracle:

```text
base+seq         0.401371
base+seq+motion  0.401757
```

So crop-motion adds only marginal headroom beyond sequence and stays an
ensemble/tie-breaker feature, not a direct submit path.

Failure explanations now supported by table evidence:

- `agn_037` test sequence has very low base/source agreement
  (`base_le15=0.4231`, `source_le15=0.5577`, `base_med_gap=358.5`), matching
  the public drop when sequence is added.
- `agn_056` validation sequence is worse despite high overlap
  (`base_le15=0.8969`, `source_le15=0.9038`, count delta `+7`), so nearest
  frame/count gates cannot fully separate sequence failures.
- `agn_062` and `agn_063` test sequence replacements still look plausible by
  overlap/count (`agn_062` source/base le15 around `0.88`; `agn_063` around
  `0.90+`), but `agn_056` shows why this remains private-risk rather than a
  public-submit trigger.

### RGB Event Filter Smoke

Created:

```text
tools/evaluate_rgb_event_filter.py
```

First smoke:

```text
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_event_filter.py \
  --predictions data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --feature-cache data/processed/rgb_features/base_resnet50_union_yolo26l_rows.npz \
  --model-name resnet50.a1_in1k --batch-size 128 --quiet
```

Result:

```text
baseline 0.365177
best tested threshold 0.05 -> 0.356329
```

This kills the simple single-frame ResNet union-crop TP/FP filter. The positive
rate of selected yolo26l events is already high (`0.919`), and the RGB filter
mostly drops true events. RGB/video should continue only as richer temporal
clip features, offset/contact witness, or segment-level scorer.

Temporal follow-up:

```text
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/evaluate_rgb_event_filter.py \
  --predictions data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --feature-cache data/processed/rgb_features/base_resnet50_union_offsets_m8_0_p8_yolo26l_rows.npz \
  --model-name resnet50.a1_in1k --frame-offsets=-8,0,8 --batch-size 192 --quiet
```

Result:

```text
best tested threshold 0.05 -> 0.351944
```

Adding temporal offsets did not fix the issue. This specifically kills the
logistic clear/drop filter target on the current selected rows. RGB may still
be useful for timing offset/contact scoring or active-exchange segment gating.

Timing-offset follow-up:

```text
.venv/bin/python tools/evaluate_rgb_timing_offset.py \
  --predictions data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --feature-cache data/processed/rgb_features/base_resnet50_union_offsets_m8_0_p8_yolo26l_rows.npz \
  --model ridge
```

Result:

```text
baseline 0.365177
best tested setting max_shift=2 scale=0.25 -> 0.357606
```

So direct RGB offset regression is also not useful on fixed selected rows.
The cached temporal RGB features may still be reusable for another target, but
not as a row filter or global offset regressor.

## Operational Notes

- Branch for this report: `research/new-growth-opportunities`.
- Existing dirty file before this report:
  `tools/make_crop_motion_context_submission.py`. It was not modified by this
  report.
- GPU 0 is busy/reserved and must remain unused. GPU 1 is the only acceptable
  target for future GPU work.
- Current `.venv` package check passed for `torch`, `torchvision`, `timm`,
  `decord`, `av`, `transformers`, `librosa`, `soundfile`, `opencv`, `sklearn`.
