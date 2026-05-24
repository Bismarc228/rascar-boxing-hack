# Candidate-Token Clear On Strict Anchor - 2026-05-23

Goal: check whether the old candidate-token clear head works better when
trained directly on the current strict `0.415261` OOF anchor, instead of
mapping stale `candidate_token_clear_*` p_keep caches from a different row
source. No Kaggle upload was made.

The old evaluator is not present in current HEAD, so this audit ran it from
history:

```text
d80c86c:tools/evaluate_candidate_token_reranker.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python <(git show d80c86c:tools/evaluate_candidate_token_reranker.py) \
  --heads clear \
  --predictions data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv \
  --tracks-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --feature-mode pose_audio \
  --epochs 30 \
  --thresholds 0.001,0.002,0.004,0.006,0.008,0.010,0.012,0.015,0.020,0.030,0.040,0.050,0.075,0.100,0.150,0.200,0.300,0.400,0.500 \
  --write-diagnostics data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv \
  --top-k 30
```

Baseline:

```text
macro=0.415261
time=0.530111
fighter=0.537953
punch_type=0.211716
effectiveness=0.289848
hand=0.521370
target=0.478143
fp_penalty=0.061715
n_pred=1188
n_tp=1096
n_fp=92
n_fn=646
```

Oracle clear/FP headroom:

```text
oracle_drop_fp_macro_delta=+0.061565
oracle_drop_fp_penalty_delta=-0.061715
oracle_time_delta=+0.000000
oracle_fighter_delta=+0.000000
oracle_effectiveness_delta=-0.001879
dropped_fp=92
dropped_tp_scorable=0
dropped_tp_time_only=0
```

The learned clear head does not rank FP rows cleanly:

```text
p_keep_tp_scorable_mean=0.8949 p50=0.9965
p_keep_tp_time_only_mean=0.8617 p50=0.9947
p_keep_fp_mean=0.8498 p50=0.9939
```

Best active threshold by macro:

```text
threshold=0.020
macro_delta=+0.001223
time_delta=-0.001233
fighter_delta=+0.001023
punch_type_delta=+0.000664
effectiveness_delta=+0.000306
hand_delta=-0.000608
target_delta=+0.000038
fp_penalty_delta=-0.001578
dropped_fp=1
dropped_tp_scorable=5
dropped_tp_time_only=2
n_pred=1180
n_fp=90
```

More aggressive thresholds improve `fp_penalty` more, but lose too much
matched evidence. For example:

```text
threshold=0.100
macro_delta=-0.007342
time_delta=-0.017616
fighter_delta=-0.014795
fp_penalty_delta=-0.007502
dropped_fp=5
dropped_tp_scorable=44
dropped_tp_time_only=3
```

Interpretation: direct current-anchor candidate-token clear training is not the
missing reproducible test path. It improves `fp_penalty` only by dropping many
more true matched rows than FP rows. The older mapped precision ranker remains
stronger on OOF (`fp_penalty_delta=-0.004533`, `macro_delta=+0.002797` for
miss top-8), but it still lacks a clean test-side generator and does not meet
the strict FP target.
