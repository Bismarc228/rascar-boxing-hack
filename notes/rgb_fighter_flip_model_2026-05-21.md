# RGB Fighter Flip Model - 2026-05-21

Goal: test whether cached row-aligned RGB clip features can recover fixed-row
fighter identity errors. This reuses the fixed-row RGB contact caches that were
already killed for keep/drop and timing, but changes the target to fighter
flips.

No Kaggle upload. No new GPU extraction was run.

## Tool

Added:

```text
tools/evaluate_rgb_fighter_flip_model.py
```

The tool:

- loads cached `N x T x D` RGB clip features aligned to fixed prediction rows;
- builds row features from clip mean/std/center/delta plus row context;
- labels OOF rows as "flip fighter" when the matched GT fighter differs within
  the scoring label window;
- trains leave-one-fight logreg/HGB classifiers and applies red/blue flips at
  confidence thresholds.

## Source

Input fixed rows:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
baseline=0.401483
fighter=0.552581
rows=1275
matched_labels=1082
flip_labels=128
```

## Union RGB Clip Cache

Cache:

```text
data/processed/rgb_features/contact_vitb16_union_clip8_stride2_current_best.npz
shape=(1275, 8, 768)
```

Best result:

```text
hgb threshold=0.30
score=0.401489
delta=+0.000006
fighter=0.552611
changed=2
```

Logreg regressed at useful thresholds; the best logreg threshold was already
negative:

```text
logreg threshold=0.95
score=0.401478
delta=-0.000006
changed=2
```

## Attacker/Opponent RGB Clip Cache

Cache:

```text
data/processed/rgb_features/contact_vitb16_attacker_opponent_clip8_stride2_current_best.npz
shape=(1275, 8, 768)
```

Best result:

```text
hgb threshold=0.30
score=0.401078
delta=-0.000405
fighter=0.550556
changed=4
```

The earlier standard threshold run also regressed:

```text
logreg threshold=0.95
score=0.401398
delta=-0.000085
changed=1
```

## Decision

Kill cached frozen RGB clip embeddings as a fixed-row fighter identity branch.
The only positive result is a two-row union-crop micro-delta (`+0.000006`),
which is below any promotion threshold and does not close the fighter oracle
gap. Do not generate a test CSV or spend a Kaggle attempt on this branch.
