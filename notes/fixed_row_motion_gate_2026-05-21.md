# Fixed-Row Motion Gate - 2026-05-21

Goal: test a lightweight RGB/crop-motion witness on fixed selected rows, without
building a full video detector and without Kaggle upload.

## Tools

Added:

```text
tools/evaluate_fixed_row_motion_gate.py
tools/make_fixed_row_motion_gate_submission.py
```

The evaluator:

- keeps row timing/fighter/attributes fixed;
- decodes only frames near selected rows;
- computes crop/global frame-diff motion at offsets `-4,-2,0,2,4`;
- concatenates those motion features with the existing pose exchange-gate
  features;
- trains a fight-group OOF keep/drop gate.

## OOF Result

Input:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_all_oof.csv
```

Baseline:

```text
score=0.400288
time=0.546891
fp_penalty=0.087252
n_rows=1282
```

Best motion gate:

```text
score=0.401483
delta=+0.001196
time=0.545894
fp_penalty=0.084963
n_rows=1275
dropped=7
threshold=0.28
```

Rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
```

Source oracle with `base/old_attr/attr_eff/seq_new/seq_motion/motion` reaches
`0.418936`; pairwise base+`seq_motion` oracle is `0.414100`.

## Test Artifacts

Full motion-gated branch:

```text
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
```

Result:

```text
Validation passed.
total_clear=743
changed_vs_root=767
```

Gated over old attribute base:

```text
submissions/hybrid_attrall_seqrepeat_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
```

Result:

```text
Validation passed.
replace_keys=agn_038,agn_047,agn_062,agn_063
total_clear=749
changed_vs_root=672
```

Public-conservative root-based no-`agn_038` splice:

```text
submissions/hybrid_root_seqrepeat_exchange_attr_motion_gate_noagn038_OFFLINE_CANDIDATE.csv
```

Result:

```text
Validation passed.
replace_keys=agn_047,agn_062,agn_063
total_clear=733
changed_vs_root=375
```

## Decision

- Keep as the best local source branch so far.
- This is a genuine new signal over the pose-only exchange gate, but the OOF
  lift is small.
- Do not upload automatically: full/gated variants still change many rows, and
  public evidence already punished full sequence replacement.
- If private-risk uploads resume, prefer the root-based no-`agn_038` splice for
  public conservation.
