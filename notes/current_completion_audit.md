# Current Completion Audit

Status as of 2026-05-21 after the post-reset public run. This is not a goal
completion claim; it records concrete evidence and remaining gaps.

## Objective Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Understand why initial score was zero | `EXPERIMENTS.md` records sample/temporal-prior public `0.00000`; metric implementation corrected the `0.5s` max timing error. | Done |
| Build local validation and metric | `rascar_boxing/metric.py`, `tools/score_predictions.py`, validation rows and OOF scores in `notes/public_lb_strategy.md`. | Done |
| Generate valid Kaggle submissions | `tools/validate_data.py` passes on root `submission.csv`; many validated candidates under `submissions/`. | Done |
| Use Kaggle submissions/leaderboard | `notes/public_lb_strategy.md` and `EXPERIMENTS.md` record public submissions, current best `0.16461`, and leaderboard check. | Done |
| Improve public score | Public moved from `0.00000` baselines to `0.16461` with `hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088`. | Done |
| Respect GPU 0 constraint | GPU work was run with `CUDA_VISIBLE_DEVICES=1`; GPU UUID check confirmed physical GPU 1 (`GPU-ff3c1fe8...`). | Done |
| Punch timing improvements | Larger pose models, sequence TCN snap4, and video-local gate are implemented and validated. Best OOF gated hybrid reached `0.395013`. | Strong progress |
| Fighter identity improvements | Whole-video swaps, cached role models, simple raw-frame ROI clustering, and ResNet50 crop clustering are killed; no robust identity fix yet. | Open |
| Audio/video feature research | Audio-only, hard snap, and direct audio rescore are killed; crop-motion has a weak independent validation gain and is saved for ensemble work. | Partially explored |
| Submission budget control | Current rule: no default uploads, at most two more attempts today, only for a clear reason above public `0.16461`. | Active guardrail |

## Current Best Artifacts

- Public-best root file: `submission.csv`.
- Public-best source:
  `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv`.
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
- Validation row tools:
  - `tools/make_pose_validation_rows.py`
  - `tools/evaluate_pose_sequence_spotter.py --write-best-rows`

## Verification Snapshot

```text
python3 -m py_compile $(find rascar_boxing tools -name '*.py' -print)  # passed
python3 tools/validate_data.py --data-root data/raw --submission submission.csv  # passed
```

Kaggle state:

```text
daily_limit=30
used_since_utc_midnight=8
remaining_today=22
current operative cap=max 2 more uploads, only with strong evidence
```

Leaderboard state:

```text
Los Pollos Hermanos  public=0.16461
next visible score   public=0.04481
```

## Remaining Gaps

- No robust fighter identity correction has passed validation. A fixed-row
  audit now shows meaningful oracle headroom (`0.389125 -> 0.402894` on the
  gated hybrid), but simple role/color flips regress. A narrow opposite-fighter
  rival rule is weakly positive (`0.389125 -> 0.389672`) and saved only as an
  ensemble micro-signal.
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
