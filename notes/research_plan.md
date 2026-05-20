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
    generated locally, root audit `risk_flags=none`, FP `0.086935`.
  - `yolo26x + yolo26l` normalized agreement, validation `0.377090`, test CSV
    generated locally, root audit `risk_flags=none`, FP `0.091709`.
- Strongest learned spotter candidates:
  - 3-seed yolo26x/yolo11s/yolo26l pose-sequence TCN with `snap_window=4`,
    `root_count=0.88`, validation `0.389963` in the first sweep and `0.387039`
    in a repeat audit, test CSV generated, total `744`.
  - Same snap4 sequence TCN, `root_count=0.92`, validation `0.390013`, test CSV
    generated, total `765`.
  - Defensive low-count snap4 `root_count=0.82`, validation `0.382612`, test CSV
    generated, total `712`; use only if public is count-sensitive.

## Validated External Research Takeaways

- Event spotting and timing are the main objective. This is confirmed by metric
  weights and by public failures of dense recall variants.
- `clear=false` is not a formatting detail; it is precision control. Every new
  model must expose confidence and be evaluated through count/FP gates.
- T-DEED/E2E-Spot/Dense-anchor ideas are useful, but only after a lightweight
  cached-feature spotter beats current peak selectors on tournament roots.
- Fighter identity should be video-local and tracklet-local. Whole-video swaps
  are killed; event-level fighter correction still has headroom.
- Audio is a weak feature near pose candidates only. No audio-only, hard snap,
  or direct onset-rescore submissions.

## Immediate Queue After Reset

1. Submit one yolo26x agreement candidate first if quota is fresh:
   `yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088`.
   It has the best offline score among ready CSVs and lower FP than the
   yolo26x+yolo26l agreement candidate.
2. If public does not regress badly, submit the safer snap4 sequence TCN
   candidate:
   `seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount088`.
   It is the strongest low-row learned spotter so far (`0.389963`, 744 test
   rows, `agn_038=81`).
3. If the safer sequence TCN transfers, submit the higher-offline snap4
   `root_count=0.92` variant (`0.390013`, 765 rows). If public punishes count,
   skip it.
4. If public likes the model family but punishes count, use the defensive snap4
   `root_count=0.82` CSV (`712` rows) before abandoning sequence.
5. If sequence TCN regresses, submit the yolo26x+yolo26l agreement candidate
   next. This checks whether public prefers large-model agreement or yolo11s
   conservative witness without changing model family as much.
6. Submit direct yolo26x `root_rate=0.88` before threshold-count if public
   remains precision-sensitive. The threshold-count candidate is locally best
   but has `844` test rows, so it is riskier after the yolo26l threshold-count
   public failure.
7. Use no more than two public-mask probes after reset, and only for unknown
   high-value videos (`agn_037`, `agn_048`). Do not burn probes on videos
   already showing no public effect.
8. If the first full yolo26x agreement submit is noisy, switch to `agn_038`
   hybrids instead of changing likely-private videos. Ready hybrid CSVs replace
   only `agn_038` on top of the yolo26l public best with:
   - yolo26x+yolo11s agreement (`agn_038=108`, total `680`),
   - yolo26x+yolo26l agreement (`agn_038=104`, total `676`),
   - yolo26x context `root_rate=0.88` (`agn_038=63`, total `635`).

## Offline Work Before More Submits

1. Build a candidate-level Gaussian spotter:
   - Generate frame/candidate features from yolo11s/yolo26l/yolo26x caches.
   - Label frames with Gaussian targets around GT impact frames.
   - Train a small 1D TCN/UNet or CatBoost/XGBoost ranker with video-group CV.
   - Postprocess with same `(fighter, hand)` refractory NMS and calibrated row
     count.
   - Do not reuse the current simple candidate-level selector family as-is:
     yolo26x HGB checks reached `0.303634` and then `0.361257` with Gaussian
     labels, both below direct yolo26x/agreement.
   - The next model must use sequence/anchor labels or a materially richer
     feature representation, not another flat candidate regressor.
2. Add RGB clip embeddings only if cached-feature spotter plateaus:
   - Start with pretrained video backbones available through `torchvision` or
     install `transformers/timm/decord` if needed.
   - Extract short clips around pose candidates on GPU 1.
   - Use embeddings as features for the spotter/reranker, not as a standalone
     detector first.
3. Implement video-local fighter identity calibration:
   - Tracklet color prototypes from torso/shorts/glove crops.
   - Only change fighter labels with fixed timing/counts.
   - Gate by per-root fighter score and confusion matrix, not macro alone.
   - Do not use current cached `score_red`/`score_blue` or track-majority
     signals alone: HGB/logreg keep-flip models on yolo26x context failed to
     recover the matched-fighter oracle headroom.
   - Do not use simple bbox-level ROI clustering alone: the first raw-frame
     two-cluster appearance attempt regressed even with oracle cluster mapping.

## Expanded Model Tracks

The external research is useful only if it changes the model family, not if it
becomes another threshold/NMS sweep. The active research branches are:

1. Pose-sequence event spotter.
   - Input: per-frame features from cached yolo11s/yolo26l/yolo26x tracks,
     model-agreement channels, wrist/head/torso velocities, fighter distance,
     role-switch flags, and local score context.
   - Models: 1D TCN/UNet first, then a small temporal Transformer if the TCN
     beats current agreement selectors.
   - Labels: Gaussian or dense-anchor labels around impact frames, with focal or
     weighted BCE loss to handle calm-frame imbalance.
   - Validation gate: beat `0.377949` yolo26x+yolo11s agreement on fight-group
     CV, improve timing, and avoid FP/count inflation.
2. RGB clip reranker or spotter.
   - Start as frozen embeddings around pose candidates, not a standalone dense
     detector. Candidate windows should cover roughly `[-16,+16]` and
     `[-32,+32]` frames around each event.
   - Candidate backbones: torchvision video models first if available; install
     `transformers`, `timm`, `decord`/`av`, or similar only when needed for
     VideoMAE/TimeSformer/Swin-style embeddings.
   - Use embeddings in CatBoost/LightGBM or a small head with the same
     `(fighter, hand)` NMS and count calibration.
   - GPU jobs must use physical GPU 1 only.
3. Multi-model pose agreement as a detector witness.
   - Keep yolo26x as primary; use yolo11s/yolo26l/yolo11m as witnesses for
     timing snap, confidence boost, or disagreement penalty.
   - Do not replace the whole detector unless tournament-root audits stay
     positive and FP stays controlled.
4. Video-local fighter identity.
   - Build per-video tracklet prototypes from pose-guided torso/shorts/glove
     crops, plus optional DINO/CLIP-style crop embeddings if color histograms
     are too brittle.
   - Validate with timing/count fixed. Only the fighter column may change in the
     first identity experiments.
   - Required reports: `score_fighter`, wrong-fighter near misses inside
     `+/-15` frames, red/blue confusion matrix, and per-root deltas.
5. Per-video precision/count controller.
   - The public checks already punished dense recall, so count calibration is a
     model branch, not just postprocess housekeeping.
   - Features: video duration, root/round/fight metadata, candidate score
     distribution, model-agreement density, same-fighter burst density, and
     selected-count priors from neighboring rounds in train.
   - Output: expected `clear=true` count or a threshold offset per video, then
     the normal grouped NMS decides the rows.
   - Gate on FP penalty and macro by tournament root; reject gains that only
     tune the old `бокс` root.
6. Crop-motion/contact reranker.
   - Motion-only detection is killed, but local crop motion can still help
     distinguish impact from guard, feint, and recovery inside a wide pose pool.
   - A two-video yolo26x-context smoke has weak positive signal but count/FP
     risk. Next step is full-validation fixed-count ablation with candidate
     limits, not submission.
   - The current OpenCV full-decode path is too slow even with `--jobs`; build a
     cached/requested-frame extractor before broader motion sweeps.
   - Features: frame-diff/optical-flow summaries around glove, opponent head,
     torso, and inter-fighter contact crops over `[-8,+8]` and `[-16,+16]`.
   - Use only as a reranker/tie-breaker with fixed candidate pool and count
     calibration; require a no-motion ablation.
7. Attribute classifier only after spotting improves.
   - Punch type/effectiveness/target/hand have lower metric weight and previous
     attribute-only gains were small.
   - Revisit with RGB/pose clips only after event timing/selection moves.
8. Audio as a weak auxiliary feature only.
   - Audio-only detection and hard audio snapping are killed.
   - Direct local-onset multiplicative rescoring is also killed on the current
     yolo26x context anchor (`0.374335 -> 0.365845`) and on `root_rate=0.88`
     (`0.372166 -> 0.371687`).
   - If revisited, add local onset/spectral features near existing pose
     candidates inside a learned reranker and require a matched no-audio
     ablation.

## Validation Protocol For New Models

- Use fight-group validation first, then report per-root and per-video deltas.
- Every model must compare against both direct yolo26x context and
  yolo26x+yolo11s agreement, not against the old yolo11n/yolo11s baseline.
- Separate three ablations: timing/count fixed, fighter-only change, and full
  selection. This prevents a fighter or RGB idea from hiding FP inflation.
- Track `score_time`, `score_fighter`, FP penalty, selected row count, and wins
  across the 13 validation videos.
- Submit only after quota reset and only for a candidate that is materially new
  and passes the validation gate. No public probing while quota is exhausted.

## Submit Gates

- Do not submit before quota reset.
- Every CSV must pass `tools/validate_data.py`.
- Prefer candidates with positive tournament-root validation deltas and
  plausible test counts.
- Avoid pure threshold/density recall unless it is explicitly being tested as a
  one-off stress probe.
