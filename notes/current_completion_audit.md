# Current Completion Audit

Status as of 2026-05-21 after the audio-gate and exchange-side identity
iterations. This is not a goal completion claim; it records concrete evidence
and remaining gaps.

## Latest Addendum - 2026-05-21

- Current best local OOF source is
  `data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_oof.csv`
  at `0.404915`.
- The first useful audio result is the fixed-row pose+audio gate:
  `0.401483 -> 0.404479`. Audio-only was neutral, pose-only was only
  `0.401999`, so the saved gain requires the learned pose+audio interaction.
- The exchange-side identity micro-model on top of the audio gate improves only
  `0.404479 -> 0.404915` and changes 22 OOF rows / 11 protected test fighter
  labels. It is saved as an ensemble/private-risk micro-signal, not a direct
  upload trigger.
- Validated full and protected test artifacts exist for both the audio gate and
  audio-gate exchange-side variants. The protected artifacts keep public-proven
  `agn_038` unchanged.
- Source-oracle diagnostics with audio raise headroom to `0.422619`; adding the
  exchange-side micro-source raises it only to `0.422744`. Learned source
  policies and stumps still fail to convert that oracle headroom into a robust
  OOF gain.
- Sequence audio-contact bridge infrastructure is implemented in
  `tools/evaluate_pose_sequence_spotter.py`, but the medium cap-400 run reached
  only `0.379792`; no test CSV was generated from that branch.
- Deep crop fighter identity calibration with ResNet50 crop embeddings regressed
  even under oracle cluster mapping, so the current unsupervised crop-cluster
  mapping path is killed.
- Submission guardrail remains unchanged: do not upload automatically; re-run
  `tools/validate_data.py` before any explicitly approved upload and avoid
  spending attempts on blind threshold/NMS/source sweeps.

## Objective Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Understand why initial score was zero | `EXPERIMENTS.md` records sample/temporal-prior public `0.00000`; metric implementation corrected the `0.5s` max timing error. | Done |
| Build local validation and metric | `rascar_boxing/metric.py`, `tools/score_predictions.py`, validation rows and OOF scores in `notes/public_lb_strategy.md`. | Done |
| Generate valid Kaggle submissions | `tools/validate_data.py` passes on root `submission.csv`; many validated candidates under `submissions/`. | Done |
| Use Kaggle submissions/leaderboard | `notes/public_lb_strategy.md` and `EXPERIMENTS.md` record public submissions, current best `0.16461`, and leaderboard check. | Done |
| Improve public score | Public moved from `0.00000` baselines to `0.16461` with `hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088`. | Done |
| Respect GPU 0 constraint | GPU work was run with `CUDA_VISIBLE_DEVICES=1`; GPU UUID check confirmed physical GPU 1 (`GPU-ff3c1fe8...`). | Done |
| Punch timing improvements | Larger pose models, sequence TCN snap4, video-local gates, crop-motion, and fixed-row pose+audio gating are implemented and validated. Current best local OOF is `0.404915`. | Strong progress |
| Fighter identity improvements | Whole-video swaps, cached role models, simple raw-frame ROI clustering, ResNet50 crop clustering, and unsupervised deep crop cluster remapping are killed. Exchange-side identity is saved as a tiny micro-signal (`0.404479 -> 0.404915`), but no robust identity fix exists yet. | Open |
| Audio/video feature research | Audio-only, hard snap, direct audio rescore, and fixed-row frozen RGB contact are killed. Fixed-row pose+audio gating is positive; sequence audio/RGB contact bridges help weak settings but are still below the best source. | Partially explored |
| Submission budget control | Current rule: no default uploads, at most two more attempts today, only for a clear reason above public `0.16461`. | Active guardrail |

## Current Best Artifacts

- Public-best root file: `submission.csv`.
- Public-best source:
  `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv`.
- Current best local OOF rows:
  `data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_exchange_side_oof.csv`
  (`0.404915`).
- Current best local non-identity source rows:
  `data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_audio_gate_oof.csv`
  (`0.404479`).
- Validated audio-gate test artifacts:
  - `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_gate_OFFLINE_CANDIDATE.csv`
  - `submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_gate_noagn038_OFFLINE_CANDIDATE.csv`
- Validated audio-gate exchange-side test artifacts:
  - `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_OFFLINE_CANDIDATE.csv`
  - `submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_noagn038_OFFLINE_CANDIDATE.csv`
- Private-risk gated candidate:
  `submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv`.
- Gate tools:
  - `tools/gate_submission_hybrid.py`
  - `tools/evaluate_gate_hybrid_rows.py`
  - `tools/analyze_gate_oracle_features.py`
  - `tools/evaluate_gate_policy_model.py`
- Identity/video tools:
  - `tools/analyze_fighter_identity_errors.py`
  - `tools/evaluate_fighter_rival_flip.py`
  - `tools/make_fighter_rival_flip_submission.py`
  - `tools/evaluate_exchange_state_gate.py`
  - `tools/make_exchange_state_gate_submission.py`
  - `tools/evaluate_fight_source_policy.py`
  - `tools/evaluate_fixed_row_attribute_model.py`
  - `tools/make_fixed_row_attribute_submission.py`
  - `tools/evaluate_deep_fighter_identity_calibration.py`
  - `tools/make_crop_motion_context_submission.py`
  - `tools/evaluate_row_source_ensemble.py`
  - `tools/evaluate_row_source_stacker.py`
  - `tools/make_relaxed_capacity_submission.py`
  - `tools/build_fight_level_source_table.py`
  - `tools/evaluate_rgb_event_filter.py`
  - `tools/evaluate_rgb_timing_offset.py`
  - `tools/evaluate_fixed_row_audio_gate.py`
  - `tools/make_fixed_row_audio_gate_submission.py`
  - `tools/make_exchange_side_fighter_model_submission.py`
- Validation row tools:
  - `tools/make_pose_validation_rows.py`
  - `tools/evaluate_pose_sequence_spotter.py --write-best-rows`

## Verification Snapshot

```text
python3 -m py_compile $(find rascar_boxing tools -name '*.py' -print)  # passed
python3 tools/validate_data.py --data-root data/raw --submission submission.csv  # passed
```

Kaggle state from the last explicit quota check; re-check before any upload:

```text
daily_limit=30
last_checked_used_since_utc_midnight=8
last_checked_remaining_today=22
current operative rule=no upload without explicit approval and strong evidence
```

Leaderboard state:

```text
Los Pollos Hermanos  public=0.16461
next visible score   public=0.04481
```

## Remaining Gaps

- The current best local OOF source is now the audio-gate exchange-side variant
  (`0.404915`), but this is not enough by itself to spend a submission. The
  public-best root remains `0.16461`, and full sequence-derived branches have
  prior public-transfer risk.
- Fixed-row pose+audio gating is the strongest new independent signal
  (`0.401483 -> 0.404479`). It has validated full and protected test artifacts,
  but the full version drops `agn_038` rows and the protected version still
  changes sequence-splice videos without fresh public evidence.
- Exchange-side fighter identity remains a micro-signal only
  (`0.404479 -> 0.404915`). It is useful for a later ensemble, not for a solo
  upload.
- Audio source-oracle headroom is real (`0.422619`, or `0.422744` with the
  exchange-side source), but current fight-level policies/stumps do not exploit
  it out of sample.
- The new audio-contact bridge is infrastructure, not a candidate yet: cap-400
  medium scored `0.379792`, below the current best. Future work should focus on
  per-video sync/latency, better candidate pools, and full validation rather
  than audio-only or global-offset submissions.
- Deep crop identity calibration is killed for the current unsupervised mapping
  design because ResNet50 crop remapping regressed even with oracle cluster
  assignment.
- No robust fighter identity correction has passed validation. A fixed-row
  audit now shows meaningful oracle headroom (`0.416256` matched-fighter oracle
  on the current best source), but simple role/color flips regress. A narrow
  opposite-fighter rival rule is weakly positive on the current best source
  (`0.401483 -> 0.401575`, 20 changed OOF rows) and saved only as an ensemble
  micro-signal.
- A learned exchange-side fighter flip classifier was tested as a high-precision
  identity model and improved only `0.401483 -> 0.401720`, still far below the
  `>=0.4045` promotion gate. It remains a diagnostic, not a submit branch.
- A fixed-row exchange/no-punch gate is now positive as a validation source:
  yolo26l base `0.365177 -> 0.370306`, sequence `0.379682 -> 0.383814`, and
  hybrid+rival `0.389672 -> 0.390962`. The validated test artifact only drops
  one row at the OOF-selected threshold, so this remains an ensemble/source
  feature rather than a direct submit trigger.
- Fight-level source policy over the current table features was tested and
  killed as a submit path. Best OOF policy (`mean_root=0.389690`) stays below
  simply using `exchange=0.390962`; Ridge/HGB policies regress.
- Fixed-row attribute modeling is now a positive independent branch:
  hybrid+rival+exchange rows improve `0.390962 -> 0.396329`, mostly through
  effectiveness. Validated public-anchor and private-risk test CSVs exist, but
  no upload was made.
- Repeating the fixed sequence TCN snap4/rootcount088 configuration and stacking
  the already positive exchange/attribute postprocess produced a new best local
  OOF source: `seq_repeat_exchange_attr_all=0.400288`. Gating that source over
  the old attribute source reaches `0.400603`, the current best local splice.
  Validated offline test CSVs exist, but the gated splice still changes `672`
  rows versus the public anchor and remains a private-risk artifact, not a
  default upload.
- A fixed-row crop-motion gate is now the best local source branch:
  `seq_repeat_exchange_attr_motion_gate=0.401483`, a `+0.001196` lift over the
  sequence-repeat exchange+attribute source. Validated full and no-`agn_038`
  test artifacts exist, but the lift is small and the branch still inherits
  sequence public-risk.
- Row-level source stacking over saved base/seq/rival/exchange/attribute/motion
  CSVs was tested and killed as a submit path. The source oracle with
  attributes reaches `0.408357`, but the learned row stacker only scored
  `0.368373`; fight-level learned policies with attribute sources also stayed
  at or below simply using `attr_all=0.396329`.
- The tabular yolo26l learned temporal selector was re-run with 16 CPU threads
  and killed for the current feature/label setup: best local score was
  `0.275094`, far below the heuristic and current ensemble sources.
- A first RGB/deep crop embedding identity smoke was tested and regressed; no
  robust RGB/video embedding model has passed validation yet.
- A first frozen RGB event-filter smoke also regressed
  (`0.365177 -> 0.356329` at the softest threshold), so single-frame ResNet
  union-crop features are not a direct filter path.
- A temporal RGB event-filter follow-up with frame offsets `-8,0,+8` also
  regressed (`0.365177 -> 0.351944`), so the current RGB clear/drop target is
  killed.
- RGB timing-offset regression over the same temporal features also regressed
  (`0.365177 -> 0.357606`), so direct RGB frame correction is not a current
  path.
- Frozen CLIP ViT-B/16 union-crop embeddings on the current
  hybrid+rival+exchange OOF source also regressed (`0.390962 -> 0.388244` at
  the best threshold), so stronger RGB work needs true clip/contact modeling or
  more structured crops.
- A structured attacker/opponent CLIP crop follow-up also regressed
  (`0.390962 -> 0.389116`), so frozen still-image CLIP filtering is killed
  more broadly.
- CLIP timing-offset regression also regressed on both union (`0.390962 ->
  0.385621`) and attacker/opponent (`0.390962 -> 0.382356`) feature caches.
- A temporal frozen ViT-B/16 fixed-row contact smoke was also tested on the
  current best source with clip embeddings (`N x 8 x 768`) and a small OOF GRU
  head. Union crop regressed to `0.385960` for keep/drop and `0.398441` for
  offset-only; attacker/opponent crop regressed to `0.398062` for keep/drop and
  `0.398528` for offset-only. This kills fixed-row frozen RGB contact as a
  direct branch; future RGB work should move to candidate-level contact or
  fine-tuning rather than another crop/mode sweep.
- Candidate-level RGB contact was tested before final NMS on a bounded raw
  yolo26x pose pool. It improves the same-pool pose baseline
  (`0.330787 -> 0.359641`) and reduces FP penalty (`0.156681 -> 0.110506`),
  but remains far below the current best local source (`0.401483`). This keeps
  RGB contact alive as an auxiliary pre-NMS signal, but not as a direct
  raw-pose row source.
- An optional RGB-contact bridge was added to `tools/evaluate_pose_sequence_spotter.py`
  so RGB contact can be blended into sequence-TCN candidate scores before final
  NMS/count. A short infrastructure smoke passed (`rgb_alpha=1.0` scored
  `0.312313` vs `0.309016` for `rgb_alpha=0.0` under a deliberately undertrained
  sequence model). A medium cap-400 run with two seeds reached `0.388112`;
  RGB helped that weaker sequence setup, but absolute OOF stayed below the
  current best `0.401483`, so no test CSV was generated.
- New source-oracle diagnostics with `exchange_side` and `seq_bridge_cap400`
  raise per-video oracle headroom to `0.420937`, but existing fight-level source
  policies still fail to exploit it (`mean_global=0.395386`, `ridge=0.393753`),
  below simply using the best single source.
- Raw crop-motion rescoring has a small independent validation gain
  (`0.365177 -> 0.368244`) and is recorded in
  `notes/ensemble_candidates.md`; the validated test artifact is
  `submissions/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008_OFFLINE_CANDIDATE.csv`.
  It is not a direct submit branch.
- The first `notes/new_growth_report_2026-05-21.md` priorities were started:
  relaxed capacity reallocation has a locally relaxed-valid diagnostic CSV, and
  fight-level source tables now exist under `data/processed/diagnostics/`.
  Neither result is a submit trigger yet.
- `.venv` is available for RGB/video/audio branches with `timm`, `decord`,
  `av`, `transformers`, `accelerate`, `librosa`, `soundfile`, and
  `open-clip-torch`; use `.venv/bin/python` plus `CUDA_VISIBLE_DEVICES=1`.
- The private-risk gated candidate has strong OOF support (`0.395013`) but is
  expected to be public-neutral beyond `0.16461`; do not upload without an
  explicit private-risk decision.
- Sequence selection still has per-video failures (`agn_056` in validation,
  `agn_037` in public). Nearest/count gate cannot fully separate these.
- Further public gains likely require either a video-local model for unknown
  public videos or a real new signal, not threshold/root-count probing.
