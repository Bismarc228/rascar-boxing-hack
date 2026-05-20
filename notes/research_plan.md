# Research Plan

Status as of 2026-05-20: Kaggle quota is exhausted (`30/30`) until
`2026-05-21T00:00:00Z`. Do not submit before reset. GPU 0 must stay unused;
all long GPU jobs use physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.

## Current Anchor

- Public best: `0.13849` from
  `yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088`.
- Strongest ready post-reset direct candidates:
  - `yolo26x_samecount_w10_am02_thr065_same8_cross4_threshold`, validation
    `0.374335`, no tournament-root risk flags, test total `844`.
  - `yolo26x_samecount_w10_am02_thr065_same8_cross4_rootrate088`, validation
    `0.372166`, no tournament-root risk flags, test total `678`.
  - `yolo26x_raw_thr065_same8_cross6_rootroundrate078`, validation `0.371622`,
    no tournament-root risk flags, test total `727`.
- Strongest offline fusion candidates:
  - `yolo26x + yolo11s` normalized agreement, validation `0.377949`, test CSV
    generated locally.
  - `yolo26x + yolo26l` normalized agreement, validation `0.377090`, test CSV
    generated locally.

## Validated External Research Takeaways

- Event spotting and timing are the main objective. This is confirmed by metric
  weights and by public failures of dense recall variants.
- `clear=false` is not a formatting detail; it is precision control. Every new
  model must expose confidence and be evaluated through count/FP gates.
- T-DEED/E2E-Spot/Dense-anchor ideas are useful, but only after a lightweight
  cached-feature spotter beats current peak selectors on tournament roots.
- Fighter identity should be video-local and tracklet-local. Whole-video swaps
  are killed; event-level fighter correction still has headroom.
- Audio is a weak feature near pose candidates only. No audio-only or hard snap
  submissions.

## Immediate Queue After Reset

1. Submit one yolo26x agreement candidate first if quota is fresh:
   `yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088`.
   It has the best offline score among ready CSVs.
2. If public does not regress badly, submit the yolo26x+yolo26l agreement
   candidate next. This checks whether public prefers large-model agreement or
   yolo11s conservative witness.
3. Submit direct yolo26x `root_rate=0.88` before threshold-count if public
   remains precision-sensitive. The threshold-count candidate is locally best
   but has `844` test rows, so it is riskier after the yolo26l threshold-count
   public failure.
4. Use no more than two public-mask probes after reset, and only for unknown
   high-value videos (`agn_037`, `agn_048`). Do not burn probes on videos
   already showing no public effect.

## Offline Work Before More Submits

1. Add a root/video audit mode for agreement candidates, matching
   `audit_pose_candidate.py`, so fusion variants are not judged by macro alone.
2. Build a hybrid/splice generator: public-tuned `agn_038` plus conservative
   offline-selected predictions for likely-private videos.
3. Build a candidate-level Gaussian spotter:
   - Generate frame/candidate features from yolo11s/yolo26l/yolo26x caches.
   - Label frames with Gaussian targets around GT impact frames.
   - Train a small 1D TCN/UNet or CatBoost/XGBoost ranker with video-group CV.
   - Postprocess with same `(fighter, hand)` refractory NMS and calibrated row
     count.
4. Add RGB clip embeddings only if cached-feature spotter plateaus:
   - Start with pretrained video backbones available through `torchvision` or
     install `transformers/timm/decord` if needed.
   - Extract short clips around pose candidates on GPU 1.
   - Use embeddings as features for the spotter/reranker, not as a standalone
     detector first.
5. Implement video-local fighter identity calibration:
   - Tracklet color prototypes from torso/shorts/glove crops.
   - Only change fighter labels with fixed timing/counts.
   - Gate by per-root fighter score and confusion matrix, not macro alone.

## Submit Gates

- Do not submit before quota reset.
- Every CSV must pass `tools/validate_data.py`.
- Prefer candidates with positive tournament-root validation deltas and
  plausible test counts.
- Avoid pure threshold/density recall unless it is explicitly being tested as a
  one-off stress probe.
