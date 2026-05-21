# Fighter Identity Error Audit - 2026-05-21

Scope: fixed selected rows only. No Kaggle upload. The goal was to separate
fighter-label errors from timing/count errors and test whether role/color or
nearby opposite-fighter evidence can safely fix labels.

## Tools Added

- `tools/analyze_fighter_identity_errors.py`
  - Reads a validation row source, matches predictions to GT, writes a row-level
    audit CSV under `data/processed/diagnostics/`.
  - Reports fixed-event `oracle_matched_fighter` headroom.
  - Scores simple track role/color/frame-color flip rules.
- `tools/evaluate_fighter_rival_flip.py`
  - Uses the raw pose candidate pool around each selected row to test whether a
    nearby opposite-fighter rival candidate should replace the row fighter.
  - Evaluates fighter-only and fighter/hand/target updates locally.
- `tools/make_fighter_rival_flip_submission.py`
  - Applies one fixed rival rule to an existing submission CSV.

## Main Audit Results

### yolo26l public-base validation rows

Command:

```bash
python3 tools/analyze_fighter_identity_errors.py \
  --predictions data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035
```

Evidence:

- Baseline: `0.365177`, fighter component `0.466772`.
- Fixed-event matched-fighter oracle: `0.382242`, fighter component `0.552098`.
- Scorable matched events: `978`.
- Fighter-wrong scorable events: `155` (`15.85%`).
- Largest wrong-fighter counts: `agn_070=20`, `agn_057=18`,
  `agn_024=16`, `agn_056=15`, `agn_069=15`.

Track/color conclusion:

- `track_role_majority`: `0.359214` (`-0.005963`), 92 flips.
- `track_color_mean`: below baseline.
- `frame_color`: `0.361077` (`-0.004100`), 75 flips.
- Strong/low-purity color gates were at best neutral or slightly negative.
- The color/role feature family is not a safe direct identity correction.

### sequence snap4 OOF rows

- Baseline: `0.379682`, fighter `0.542061`.
- Fixed-event matched-fighter oracle: `0.398103`, fighter `0.634162`.
- Fighter-wrong scorable events: `144/1087` (`13.25%`).
- Simple role/color flips again did not improve.

### gated yolo26l/sequence hybrid OOF rows

- Baseline: `0.389125`, fighter `0.510594`.
- Fixed-event matched-fighter oracle: `0.402894`, fighter `0.579439`.
- Fighter-wrong scorable events: `120/1026` (`11.70%`).
- Simple role/color flips again did not improve.

## Opposite-Fighter Rival Candidate Test

Hypothesis: some fighter errors are not equipment/track-color mistakes; they
are exchange-side mistakes where an opposite-fighter pose candidate near the
same frame is stronger.

Results:

| Row source | Best local score | Delta | Best rule | Changed rows | Notes |
| --- | ---: | ---: | --- | ---: | --- |
| yolo26l public-base OOF | `0.365447` | `+0.000270` | `window=4`, `same_hand_target`, `ratio=1.3`, fighter-only or fighter/hand/target | 3 | Weak but positive. |
| sequence snap4 OOF | `0.380125` | `+0.000443` | `window=2`, `same_hand_target`, `ratio=1.6`, fighter-only or fighter/hand/target | 14 | Weak but positive. |
| gated hybrid OOF | `0.389672` | `+0.000548` | `window=0`, `same_hand`, `ratio=1.3`, `min_rival_score=1.0`, fighter/hand/target | 8 | Best weak ensemble branch. |

Rival diagnostics show why this must stay gated: nearby rivals would fix many
wrong rows, but would break far more correct rows if applied broadly. The useful
configuration is extremely narrow and high precision.

## Test Artifact

Generated from current public-best root `submission.csv`:

```bash
python3 tools/make_fighter_rival_flip_submission.py \
  --input submission.csv \
  --output submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_w0_samehand_r13_min1_fht_OFFLINE_CANDIDATE.csv \
  --tracks-dir data/processed/pose_tracks/test_yolo26x_conf035 \
  --window 0 \
  --match-mode same_hand \
  --ratio 1.3 \
  --min-rival-score 1.0 \
  --update-mode fighter_hand_target
```

Validation:

```text
python3 tools/validate_data.py --data-root data/raw \
  --submission submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_w0_samehand_r13_min1_fht_OFFLINE_CANDIDATE.csv
Validation passed.
```

Changed rows: 7. Affected videos: `agn_038`, `agn_048`, `agn_049`,
`agn_062`, `agn_063`.

## Ensemble Check

Generated validation row artifacts:

- `data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv`
- `data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_rival_w4_samehandtarget_r13.csv`

Source ensemble run:

```bash
python3 tools/evaluate_row_source_ensemble.py \
  --source base=data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv \
  --source seq=data/processed/validation_rows/seq_tcn_snap4_rootcount088_oof.csv \
  --source hybrid=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_oof.csv \
  --source hybrid_rival=data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_w0_samehand_r13_min1_fht.csv \
  --source motion=data/processed/validation_rows/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008.csv
```

Result:

- `hybrid_rival`: `0.389672` versus `hybrid`: `0.389125`.
- Pairwise base+hybrid_rival oracle: `0.393859` versus base+hybrid oracle
  `0.393346`.
- Full source oracle across base/seq/hybrid_rival/motion: `0.402103`.
- Best-source wins unique to `hybrid_rival`: `agn_024`, `agn_025`,
  `agn_070`.

This confirms the branch is a small independent identity/attribute signal, not
just a duplicate of sequence or motion.

## Decision

- Do not submit this branch directly. The OOF gain is too small for the current
  submit guardrail.
- Keep it in the ensemble registry as an independent high-precision identity
  micro-signal.
- Do not spend more time on global role/color flips unless new mask/tracklet
  features change the evidence.
- Next identity work should focus on exchange-side classification or persistent
  tracklet/mask identity, not simple equipment color.
