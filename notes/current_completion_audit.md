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
| Fighter identity improvements | Whole-video swaps, cached role models, and simple raw-frame ROI clustering are killed; no robust identity fix yet. | Open |
| Audio/video feature research | Audio-only, hard snap, and direct audio rescore are killed; crop-motion smoke has weak signal but needs better cached-frame extractor. | Partially explored |
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

- No robust fighter identity correction has passed validation.
- No RGB/video-embedding model has been tested yet; only pose, audio, and
  light crop-motion branches were explored.
- The private-risk gated candidate has strong OOF support (`0.395013`) but is
  expected to be public-neutral beyond `0.16461`; do not upload without an
  explicit private-risk decision.
- Sequence selection still has per-video failures (`agn_056` in validation,
  `agn_037` in public). Nearest/count gate cannot fully separate these.
- Further public gains likely require either a video-local model for unknown
  public videos or a real new signal, not threshold/root-count probing.
