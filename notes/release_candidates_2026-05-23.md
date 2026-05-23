# Release Candidates - 2026-05-23

Goal: name and document three internal release candidates before deciding what
to promote.

Metrics below are local internal validation/OOF metrics on 13 validation
videos, using the repository metric implementation. Higher is better. Component
columns are macro-averaged per-video component scores.

## RC1: Shadow Stack ClearCut

Artifact:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Internal metric source:

```text
data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv
data/processed/diagnostics/clear_ranked_on_private_stack_transitionopt_thresholds_20260523.csv
```

Short description:

Private-stack attribute candidate on top of the strict public-best family. It
keeps the fixed row/timing structure, adds a transition-optimized punch-type
component, DINO ROI effectiveness corrections, a target guard, then applies a
small high-confidence clear-drop rule over blocked/miss rows.

Main features:

- Fixed-row stack: timing/count mostly inherited from the strict anchor.
- DINO ROI effectiveness signal, focused on landed-to-nonlanded corrections.
- Transition-optimized punch-type replacement.
- Target guard with small positive local target lift.
- `top8 blocked_miss` token-clear step drops the eight riskiest blocked/miss
  rows by `p_keep`, trading a small time loss for lower FP penalty.

Internal score:

```text
macro_score     0.420768
n_pred          1180
n_tp            1091
n_fp            89
```

Component breakdown:

```text
time            0.529325
fighter         0.538783
punch_type      0.224796
effectiveness   0.311123
hand            0.521381
target          0.481339
fp_penalty      0.059184
```

Build-up:

```text
strict_base                         0.415261
private_stack transitionopt guard   0.418474
token_clear top8 blocked_miss       0.420768
```

Notes:

The final clear-drop step improves local score mainly through FP penalty:
`fp_penalty_delta=-0.002531`, while time moves down slightly
`time_delta=-0.000785`. Public LB for this exact submitted file was `0.19069`.

## RC2: Old-Attribute Source Switch

Artifact:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
```

Related test analog:

```text
submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

Short description:

Fight-level source-switch candidate. It starts from the newer sequence motion
attribute source and switches selected videos back to the older attribute source
when the old source has strong frame-neighborhood agreement with the base
source.

Main features:

- Base source: `seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate`.
- Override source: `hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all`.
- Switch rule: use old attribute source when `old_attr.source_le15 >= 0.93`.
- Selected validation videos: `agn_003`, `agn_024`, `agn_025`, `agn_056`,
  `agn_057`, `agn_058`, `agn_069`, `agn_070`, `agn_071`.
- Designed as a source-policy/ensemble release rather than a single model.

Internal score:

```text
macro_score     0.410780
n_pred          1263
n_tp            1144
n_fp            119
```

Component breakdown:

```text
time            0.541274
fighter         0.549713
punch_type      0.198453
effectiveness   0.260609
hand            0.526352
target          0.491679
fp_penalty      0.071576
```

Build-up:

```text
seq_motion base       0.401483
source switch         0.410780
delta                +0.009297
```

Notes:

The local gain is large but concentrated: without `agn_003`, the switch regresses
from `0.419714` to `0.418878`. Group-OOF stump validation scored `0.399638`,
so this is a high-internal-score release candidate with notable overfit risk.

## RC3: Candidate Token Punch-Type Mapper

Artifact:

```text
data/processed/validation_rows/candidate_token_punch_type_margin095_20260522_oof.csv
```

Short description:

Punch-type replacement candidate based on candidate-token class probabilities.
It keeps the event rows fixed and changes punch type where the token model has a
high-margin alternative prediction.

Main features:

- Fixed-row OOF candidate: no row-count or timing expansion from the token model.
- Focused on punch-type relabeling rather than detection.
- Margin-gated token prediction with `margin095` source.
- Useful as an attribute component or ensemble input, even though direct
  strict-anchor replacement is not the strongest punch-type path.

Internal score:

```text
macro_score     0.408187
n_pred          1226
n_tp            1113
n_fp            113
```

Component breakdown:

```text
time            0.537088
fighter         0.544665
punch_type      0.213339
effectiveness   0.278732
hand            0.526763
target          0.485253
fp_penalty      0.073644
```

Attribute audit on strict anchor:

```text
n_changed                509
strict_anchor_macro      0.415261
token_ptype_macro        0.415162
macro_delta             -0.000099
score_punch_type_delta  -0.000993
```

Notes:

As a standalone validation-row source it scores `0.408187`. As a pure
punch-type replacement on the strict anchor it is slightly negative, so the
release value is mainly as a candidate-token attribute source for downstream
stacking, ablations, and ensembling.
