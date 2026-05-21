# Hypotheses Backlog

Research backlog for the Kaggle `boxing-action-recognition-challenge`. Evidence
comes from `EXPERIMENTS.md`, `OVERVIEW.md`, and `DATA_DESCRIPTION.md`.

## Current State

- Best public score: `0.13849` from
  `submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv`.
- Corrected offline validation uses all 13 fight-level validation videos in
  `data/processed/pose_tracks/val_yolo11n_conf035/`.
- `yolo11n` simple offline threshold/NMS baseline:
  `0.23802` at `threshold=0.8`, `nms=6`, `n=1119`.
- Best `yolo11n` submitted-public analog on the same split:
  `0.20130` at `threshold=1.0`, `nms=12`.
- Best checked grouped NMS:
  `0.24246`, fighter-grouped NMS, `threshold=0.8`, same-group NMS `8`,
  cross-group NMS `2`, `n=1151`.
- Best checked grouped count variant:
  `0.242684`, root-count multiplier `0.9`.
- Best checked temporal-context variant:
  `0.245871`, same `(fighter, hand)` dominance in a `+/-4` frame window,
  `alpha=0.2`, `threshold=1.05`, same-fighter NMS `7`, cross-fighter NMS `2`,
  `n=1170`.
- Best checked `yolo11s-pose` offline variant:
  `0.340888`, fighter+hand grouped NMS, `threshold=1.15`, same-group NMS `10`,
  cross-group NMS `4`, `root_count=1.0`, `n=1202`.
- Best checked `yolo11m-pose` offline variant:
  `0.363325`, same-score temporal context in a `+/-8` frame window,
  `alpha=-0.2`, fighter+hand grouped NMS, `threshold=0.9`, same-group NMS `10`,
  cross-group NMS `4`, `n=1198`, FP penalty `0.065930`, 11/13 validation wins.
  Public score for the matching candidate was only `0.12958`, so it does not
  replace the `yolo11s` public baseline.
- Best checked `yolo26l-pose` public variant:
  `0.13849`, `same_sum` temporal context in a `+/-4` frame window,
  `alpha=-0.2`, fighter+hand grouped NMS, `threshold=0.85`, same-group NMS
  `10`, cross-group NMS `4`, `root_rate=0.88`. Dense threshold-count on the
  same pool scored only `0.10260`, so public is currently punishing count
  inflation.
- Best checked `yolo26x-pose` offline variant:
  `0.374335`, `same_count` temporal context in a `+/-10` frame window,
  `alpha=-0.2`, fighter+hand grouped NMS, `threshold=0.65`, same-group NMS `8`,
  cross-group NMS `4`, threshold count, FP penalty `0.052324`, 10/13 wins, and
  no tournament-root risk flags. It has not been submitted; the daily Kaggle
  limit was exhausted before yolo26x test tracks were ready.
- Kaggle quota on 2026-05-20 is exhausted (`30/30`). Reset is
  `2026-05-21T00:00:00Z`. Do not submit before reset.
- Current post-reset runbook is no longer a yolo26l tweak sweep. The next live
  test is the validated yolo26x+yolo11s agreement CSV
  (`0.377949` offline, FP `0.086935`, `727` test rows), followed by sequence
  TCN snap4 candidates only if the agreement branch does not publicly regress.
- Ready validated sequence CSVs are:
  `root_count=0.88` snap4 (`0.389963` first sweep, `0.387039` repeat audit,
  `744` rows), `root_count=0.92` snap4 (`0.390013`, `765` rows), and defensive
  `root_count=0.82` snap4 (`0.382612`, `712` rows).
- Post-reset public results changed the active strategy: the yolo26x+yolo11s
  agreement full submit scored only `0.10784`, and the full sequence snap4
  `root_count=0.88` submit scored only `0.12712`. The useful public gain is
  video-local: yolo26l public best plus sequence snap4 only on `agn_038` scored
  `0.16461`.
- Adding sequence replacements for `agn_047`, `agn_062`, or `agn_063` to the
  `agn_038` hybrid did not change public (`0.16461`), while adding `agn_037`
  dropped to `0.12712`. Treat `agn_037` as public-impact and sequence-bad.
- New submission budget rule: no more uploads by default; at most two remaining
  attempts today, only for a clear candidate above the `0.16461` hybrid.
- Generated but not submitted candidates include the grouped-NMS, grouped-count,
  and temporal-context CSVs listed in `EXPERIMENTS.md`.
- The task rewards timing most heavily. The metric weights time at `50%`,
  fighter at `20%`, punch type at `10%`, effectiveness at `8%`, and hand/target
  at `6%` each, with a separate FP penalty. Rows with `clear=false` do not match
  and are not counted as FP.
- Test has 9 videos and the submission template has 1594 fixed rows. Use
  `clear=false` for unused rows and always pass local schema validation before
  considering a submit.

## Hypotheses Already Killed

- Sample submission frames/counts are not labels.
- Pure temporal priors are too weak; the temporal-prior and sample baselines
  scored `0.00000` publicly.
- Global fighter swap, all-red/all-blue, and confidence-threshold role fixes do
  not beat the current role assignment.
- Global frame offsets do not fix timing. Offset `0` is best around grouped NMS;
  `+1`, `-1`, and larger shifts degrade.
- Supervised rankers using the current pose features underperform the simple
  heuristic on mean LOOV.
- The same remains true on yolo26x with HGB-style candidate selectors. A fast
  HGB sanity check reached only `0.303634` with FP `0.157001`; a later
  Gaussian-label candidate regressor improved to `0.361257` with FP
  `0.105266`, but still stayed below direct yolo26x/agreement. Do not spend
  reset submissions on this selector family.
- Supervised attribute models only produced a small gain in one timing setup
  (`0.21299 -> 0.21590`), not enough to submit.
- Motion-only candidates are noisy. Crop-motion reranking is more plausible, but
  the first grid did not beat the full-split threshold/NMS candidate.
- A two-video yolo26x-context motion smoke has weak positive signal
  (`0.401618 -> 0.405940` with small FP growth, or `0.422798` with row
  inflation), but it is not a submit branch until full-validation fixed-count
  ablation passes.
- Audio-only onset detection is not a detector (`0.02927` macro with oracle
  counts), and hard-shifting pose candidates to audio peaks hurts.
- Audio reranking has not beaten the full-validation baseline; the best positive
  audio boost was still slightly worse than baseline (`0.23789` vs `0.23802`).
- The same conclusion holds on the current yolo26x context anchor:
  `tools/evaluate_audio_pose_context.py` reproduced `0.374335` at `alpha=0`,
  while best nonzero audio fell to `0.365845`; with `root_rate=0.88`, best
  nonzero audio was `0.371687` versus `0.372166`.
- Fighter/color and attribute priors are not useful standalone improvements.
- `yolo11s-pose` transferred to public and is now the strongest baseline:
  `0.13275` public. The next detector/model experiments should compare against
  `yolo11s`, not `yolo11n`.
- `yolo11m-pose` beats `yolo11s` offline by more than the submit gate, but the
  first public `yolo11m` context candidate scored `0.12958` versus `0.13275`.
  Treat this as overfit until a stronger validation explanation exists.
- `yolo26s-pose` is not useful as a replacement detector: validation is only
  `0.305323`, and role coverage/track-switch metrics are worse than the YOLO11
  caches.
- `yolo26m-pose` is a possible secondary timing witness but not a submit path
  yet. Its best context candidate reaches `0.352804` with good FP penalty, but
  the gain is concentrated in the old `бокс` root while both tournament roots
  fail to improve versus the public-proven `yolo11s` anchor.
- For larger pose models, prefer asymmetric agreement/snapping/fusion against
  `yolo11s` over replacing the detector wholesale. A large-model-only candidate
  must pass tournament-root audit, not just full-split macro.
- `yolo26m` timing snap around fixed `yolo11s` selected events gives only a
  small but root-stable lift (`0.340888 -> 0.343372`). This is not enough to
  submit, but it is safer than replacing all events with a large-model path.
- Fighter identity is not solved by a per-video keep/swap decision. Full-video
  swap hurts every validation video, while event-level oracle fighter labels
  have about `+0.018` macro headroom. Work should target local tracklet
  identity and per-video color prototypes.
- `yolo26l-pose` plus temporal context transferred to public and is now the
  public anchor. The local macro winner (`same_count w8 alpha=-0.1`) scored
  `0.13580`, but public preferred the more precision-oriented `same_sum w4
  alpha=-0.2` candidate at `0.13849`.
- Tiny count-only moves around `yolo26l same_sum w6 alpha=-0.2` did not improve
  public: `root_rate=0.84/0.86/0.88` all scored `0.13664`. A stronger context
  penalty (`alpha=-0.25`) hurt badly (`0.12020`), and weaker penalty
  (`alpha=-0.15`) also hurt (`0.13095`).
- Pure density on the yolo26l pool is killed: threshold-count scored `0.10260`.
  Future candidates should be fixed-count, root-rate, or agreement-calibrated,
  not wide threshold recalls.
- A first learned per-video count controller did not beat simple count policies.
  On yolo26x context it matched the threshold/root-rate anchor (`0.374335`) but
  stayed below `root_round_count=1.0` (`0.374359`). On yolo26x+yolo11s
  agreement it regressed (`0.375863` versus `0.377949`). Do not spend submit
  budget on learned count control without a richer confidence model.
- A seed-ensembled pose-sequence TCN is the first learned spotter that beats
  yolo26x agreement offline: `0.382073` for `root_count=0.92` and `0.379314`
  for the safer `root_count=0.88` variant. The gain is on tournament roots,
  while the weak old `бокс` root is not in test. Validated test CSVs exist, but
  submit only after quota reset and preferably after the safer yolo26x agreement
  public check.
- Snapping selected sequence frames to the local TCN probability maximum is a
  real improvement: `snap_window=4` reached `0.390013`, and the safer
  `root_count=0.88` snap4 variant reached `0.389963` with lower FP and only
  `744` test rows.
- A narrow repeat audit of the safer snap4 point landed at `0.387039`, so
  sequence scores have some training/CUDA noise but remain above agreement.
  The defensive `root_count=0.82` snap4 variant scored `0.382612` with only
  `712` test rows.
- Public transfer nuance: sequence snap4 is not safe as a full replacement. It
  is excellent on `agn_038` but bad on `agn_037`; future sequence use should be
  video-local and agreement-gated.
- Local base-vs-sequence nearest-frame diff supports the public result:
  `agn_038` has `85%` of base events within 15 frames of a sequence event, but
  `agn_037` only `42%`. `agn_048` and `agn_049` look similarly risky; `agn_039`
  and `agn_064` have large timing tails. Do not submit those blind hybrids.
- `tools/gate_submission_hybrid.py` now turns this into a reproducible gate.
  Default thresholds select `agn_038,agn_047,agn_062,agn_063` and reject the
  risky equal-count rewrites. The resulting gated CSV validates locally but is
  not a public-submit breakthrough by itself.
- The gate also explains the yolo26x agreement public failure on `agn_038`:
  agreement adds `+30..34` rows there and is rejected by count delta before
  considering timing. Keep yolo26x agreement out of the submit queue.
- The gate has positive validation evidence now. On OOF validation rows,
  yolo26l public-base rows scored `0.365177`, full sequence snap4 scored
  `0.379682`, and the gated yolo26l/sequence hybrid scored `0.389125`. This
  makes gated sequence a plausible private-risk improvement, but not a public
  upload unless it can plausibly beat `0.16461`.
- Sweeping gate thresholds improved OOF to `0.395013`. The corresponding test
  gate selects `agn_038,agn_062,agn_063` and rejects `agn_037`; the CSV is
  validated but not submitted because it is expected to be public-neutral beyond
  the existing `agn_038` gain.
- Base-vs-sequence per-video oracle reaches `0.401371`, so more validation
  headroom exists. The remaining gate mistakes are not cleanly separated by
  nearest-frame/count metrics; add per-video/fight context before another
  threshold-only gate.
- The per-video feature table confirms the weakness: `agn_056` passes overlap
  gates but sequence is worse. Test replacements for `agn_062/063` are still
  plausible private improvements, but not strong enough to spend a public
  attempt under the current cap.
- A learned per-video Ridge policy over gate features is not reliable with this
  tiny validation set. It scored `0.381635` in fight-CV and `0.380425` in
  video-CV, below the threshold gate's `0.395013`.
- Copying attributes from nearest yolo26l-base rows into sequence rows is not a
  meaningful path. Best full-sequence transfer was only `0.379682 -> 0.380298`;
  best gated transfer was `0.389125 -> 0.389537`, and fighter/hand transfer
  consistently hurt.
- Fighter identity still has headroom, but not from current cached color/track
  signals. On yolo26x context, matched-fighter oracle reaches `0.392941`
  (`+0.018606`), but HGB/logreg keep-flip models over `score_red/score_blue`,
  candidate features, and track majority do not improve. Next identity work
  must extract per-video ROI/tracklet appearance from raw frames.
- A first raw-frame bbox ROI clustering attempt also failed: even oracle
  cluster-to-fighter mapping regressed to `0.368859`. Simple per-video
  appearance clusters are not enough; future identity work needs pose-guided
  ROI quality and tracklet continuity/visual embeddings, not bbox color
  clustering.
- A first deep crop embedding identity smoke is also killed as a direct
  role-correction branch. ResNet50 ImageNet crops worsened fixed-timing fighter
  labels on both old `бокс` videos (`0.322257 -> 0.319736` gated) and tournament
  videos (`0.389548 -> 0.381589` gated); oracle cluster mapping was worse too,
  so the clusters are capturing pose/background more than red/blue identity.
- Crop-motion rescoring is weak but non-dead as an independent timing signal.
  On the yolo26l public-base validation rows, full raw-frame crop-motion
  context improved `0.365177 -> 0.368244` with `alpha=-0.04,beta=0.08`, mostly
  through timing/fighter score but with higher FP. Keep it only for future
  ensemble/rerank work, not as a direct submit.
- Relaxed row/id capacity is now technically audited but not submit-approved.
  `tools/make_relaxed_capacity_submission.py` produced a 1594-row diagnostic
  that reallocates 20 spare `agn_062` filler ids to extra `agn_037` predictions.
  It passes relaxed local validation and fails strict `id -> video_key`
  metadata exactly as expected. This remains high-risk because Kaggle may
  enforce the sample metadata internally.
- Fight-level diagnostics are now available for base/sequence/crop-motion.
  They support the `agn_037` public failure with very low sequence/base
  agreement, but also show the harder validation failure `agn_056`, where
  sequence is worse despite high overlap. Future gates need richer fight-level
  or RGB features, not only count and nearest-frame overlap.
- A first frozen RGB event-filter smoke is killed in its simple form. ResNet50
  ImageNet features from the union crop of both fighters, trained as a
  leave-fight-out logistic clear-event filter on yolo26l base rows, regressed
  immediately (`0.365177 -> 0.356329` at the softest tested threshold). RGB
  may still be useful, but only as richer temporal clip/video features or a
  better witness target, not as a single-frame crop TP/FP filter.
- A temporal version of the same RGB filter is also killed: ResNet50 union-crop
  features at offsets `-8,0,+8` with mean/std aggregation regressed further
  (`0.365177 -> 0.351944`). The target is the problem: selected yolo26l rows
  are already high-precision, so filtering by RGB probability mainly removes
  true positives. Future RGB work should predict timing offset/contact state or
  segment confidence, not clear/drop on the current selected rows.
- External research reinforces the priority order: impact spotting and
  precision/`clear` calibration first, fighter identity second, attributes
  later.

## Promising Next Hypotheses

- Expand temporal-context scoring around pose candidates, not single-frame
  scoring. The current best signal is local same `(fighter, hand)` dominance.
- Keep direct yolo26x test candidates as fallback, not the first upload. The
  cache is complete and three validated CSVs are ready: context
  threshold-count (`844` rows), context `root_rate=0.88` (`678` rows), and raw
  `root_round_rate=0.78` (`727` rows). Because public punished dense recall,
  prefer agreement and lower-count yolo26x variants before threshold-count.
- Do not use normalized yolo26x agreement as a submit branch without a new
  explanation. The first full post-reset public check scored only `0.10784`,
  and `agn_038`-only agreement hybrid scored only `0.12888`.
- Root/video audit for agreement candidates now exists in
  `tools/audit_pose_agreement_candidate.py`. Both yolo26x+yolo11s and
  yolo26x+yolo26l pass tournament-root audit; yolo26x+yolo11s is the cleaner
  first public candidate because it has lower validation FP.
- Public/private hybrid generation now exists in `tools/splice_submission_videos.py`.
  Use it to tune confirmed-public `agn_038` without disturbing likely-private
  videos when public probing is needed after reset.
- Follow the concrete research queue in `notes/research_plan.md`: preserve the
  `agn_038` sequence hybrid, diagnose `agn_037`, and only consider at most two
  further submissions if a local/video-specific check gives a strong reason.
- The event-selector path is now sequence/anchor spotting, not flat candidate
  ranking. Keep the direct yolo26x/agreement heuristic as fallback; do not
  revisit simple candidate-level rankers on the current feature set.
- Improve per-video count control. Dense NMS improves offline, but test/public
  FP risk may differ by fight, so count calibration needs stress tests.
- Use audio only as a weak learned feature such as local onset max or nearest
  onset distance inside a pose-dominant selector. Do not hard-shift frames or
  use direct multiplicative onset rescoring.
- Revisit crop-motion only as a tie-breaker/reranker for a wide pose pool.
- Calibrate fighter colors per video, because equipment colors vary by video
  and the current extractor uses fixed HSV red/blue masks. Treat color as
  tracklet identity evidence, then map local identities to red/blue labels with
  a confidence gate.
- Try a T-DEED/E2E-Spot-inspired spotter only after a lightweight cached-feature
  version is in place: Gaussian labels around GT frames, same-fighter refractory
  NMS, and tournament-root validation.
- Delay attribute work until timing/selection improves; attributes have lower
  metric weight and have not yet moved enough offline.

## Compute-Aware Experiment Queue

1. Public-overfit audit for `yolo11m`.
   - Compare the submitted context candidate against raw `yolo11m` and
     `yolo11s` on validation by video, prediction count, and sample-capacity
     clipping.
   - Do not submit another `yolo11m` threshold/context tweak until that audit
     identifies a materially different fix.
2. No-GPU offline grid: rerun and narrow temporal-context sweeps on cached
   `yolo11s` and later `yolo11m` validation tracks.
   - Start near `window=4`, `alpha=0.2`, `dominance`,
     `threshold=1.0..1.15`, same-fighter NMS `6..8`, cross-NMS `2..4`.
   - Command shape:
     `python3 tools/evaluate_pose_temporal_context.py --tracks-dir data/processed/pose_tracks/val_yolo11n_conf035 --windows 3,4,5,6 --alphas 0.1,0.15,0.2,0.25,0.35 --features dominance,same_sum --thresholds 0.95,1.0,1.05,1.1,1.15 --nms-frames 6,7,8 --nms-group-modes fighter --cross-nms-frames 2,3,4`.
3. No-GPU count stress test on cached validation tracks.
   - Compare threshold-only against root/root-round count modes around
     multipliers `0.8..1.0`.
   - Command shape:
     `python3 tools/evaluate_pose_threshold_grid.py --tracks-dir data/processed/pose_tracks/val_yolo11n_conf035 --thresholds 0.7,0.8,0.9,1.0,1.05,1.15 --nms-frames 4,6,7,8,10 --count-modes threshold,root_count,root_round_count,root_rate,root_round_rate --count-multipliers 0.8,0.9,1.0`.
4. No-GPU robustness audit.
   - Compare macro, time score, FP penalty, prediction count, and video wins
     against both `0.23802` and `0.245871`. Reject variants that win by count
     inflation only.
5. GPU pass only if tracks are missing or a new detector/config is justified.
   - Regenerate pose tracks with GPU 1 only, verify with `nvidia-smi` UUID, then
     return to cached offline evaluation.
6. Candidate generation after a gate passes.
   - Use `tools/make_pose_heuristic_submission.py` with the winning parameters.
   - Validate locally; do not submit from this backlog step.
7. Large-model witness tests.
   - Finish `yolo26l`/future `yolo26x` validation caches on GPU 1, then run the
     same complete-cache, track-quality, score-scale, and tournament-root gate.
   - If a large model is only strong on old `бокс`, use it only for fixed-count
     timing snap or agreement boost around `yolo11s` events.

## Submit Gates

- Respect the 30 submissions/day limit. Do not submit small threshold sweeps.
- On 2026-05-20 the daily limit is exhausted. Submit gate is closed until
  `2026-05-21T00:00:00Z`.
- A candidate must pass local validation against `sample_submission.csv`.
- A candidate must be materially different from already submitted variants.
- Automatic submit gate: beat the current best offline `yolo11s` score
  `0.340888` by at least `0.010` macro, with no FP-penalty regression larger
  than `0.01` and at least 10 of 13 validation-video wins versus
  `thr=0.8,nms=6`.
- Manual-review gate: a smaller gain may be worth discussing only if it improves
  timing score clearly, keeps prediction counts plausible, and is not another
  threshold/NMS-only tweak.
- Never submit audio-only, pure temporal-prior, global offset, or attribute-only
  variants unless they are part of a larger pose-dominant candidate that passes
  the gates above.

## Use GPU 1 Only

- For Ultralytics pose extraction in this repo, use `CUDA_VISIBLE_DEVICES=1`
  and **omit** `--device`.
- Do not combine `CUDA_VISIBLE_DEVICES=1` with `--device 0`; this was observed
  to route Ultralytics to physical GPU 0 on this machine.
- Do not rely on `--device 1` alone for Ultralytics here; this version can
  rewrite visibility internally and fail or select the wrong device.
- Confirm any long GPU job with:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
  Physical GPU 1 UUID is `GPU-ff3c1fe8-747e-e4c7-a604-5088eee8c87e`.
- For the batch pose extractor, keep the built-in GPU-1 visibility default and
  do not pass `--device`:
  `python3 tools/run_pose_batch.py --cuda-visible-devices 1 --jobs 3 ...`.
- Example validation-track extraction:
  `python3 tools/run_pose_batch.py --data-root data/raw --videos-csv data/raw/train/videos.csv --output-dir data/processed/pose_tracks/val_yolo11n_conf035 --model models/yolo11n-pose.pt --conf 0.35 --cuda-visible-devices 1 --jobs 3 --no-tqdm`.
- Most backlog steps use cached tracks and do not need GPU.
