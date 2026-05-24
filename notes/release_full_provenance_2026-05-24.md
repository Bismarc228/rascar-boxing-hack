# Release CSV Provenance - 2026-05-24

Goal: make the two release candidates auditable below their ignored CSV inputs.
The release branches must not rely on "some CSV exists locally" as unexplained
state. This note is the index for the code and notes now carried in the branch.

Large artifacts are still intentionally ignored by Git:

- YOLO weights: `*.pt`
- generated submissions: `submissions/`
- pose tracks, feature caches, diagnostics, validation rows: `data/processed/`

Those files are too large or too volatile to commit. The branch instead carries
the code, exact commands, provenance notes, and a machine-checkable manifest
required to rebuild and verify them:

```text
release_artifacts/release_artifacts_manifest.tsv
tools/bootstrap_release_assets.py
tools/verify_release_artifacts.py
notes/release_reproducibility_2026-05-24.md
requirements-release.txt
```

## Current Hard Boundary

The remaining non-source inputs are:

```text
data/raw/
models and root-level YOLO *.pt weights
HuggingFace/timm model caches for CLIP, VideoMAE, and DINOv2
generated feature caches under data/processed/
```

They should not be silently treated as source code. Use the manifest to verify
local caches by SHA-256, and use `tools/bootstrap_release_assets.py` to restore
external model weights without committing them.

## RC1: Shadow Stack ClearCut

Final runner:

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py
```

Final output:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

### RC1 DAG

```text
raw videos
  -> YOLO pose tracks
  -> CLIP/VideoMAE feature caches
  -> strict fixed-row public-best anchor
  -> private component stack:
       DINO ROI effectiveness guard
       root-out punch_type transition guard
       target guard
  -> candidate-token clear p_keep
  -> top8 blocked/miss clear drop
```

### Pose Weights and Tracks

The final RC1 clear model uses `yolo26l` pose/audio features:

```text
yolo26l-pose.pt
data/processed/pose_tracks/val_yolo26l_conf035/
data/processed/pose_tracks/test_yolo26l_conf035/
```

Track generation command shape:

```bash
python3 tools/run_pose_batch.py \
  --data-root data/raw \
  --videos-csv data/raw/train/videos.csv \
  --output-dir data/processed/pose_tracks/val_yolo26l_conf035 \
  --model yolo26l-pose.pt \
  --conf 0.35 \
  --cuda-visible-devices 1 \
  --jobs 1 \
  --no-tqdm

python3 tools/run_pose_batch.py \
  --data-root data/raw \
  --videos-csv data/raw/test/videos.csv \
  --output-dir data/processed/pose_tracks/test_yolo26l_conf035 \
  --model yolo26l-pose.pt \
  --conf 0.35 \
  --cuda-visible-devices 1 \
  --jobs 1 \
  --no-tqdm
```

### Strict Anchor Below RC1

The fixed-row anchor used by the release is:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Its recovery path is documented in:

```text
notes/rebuild_clearens_test_artifact_2026-05-22.md
notes/combined_fixed_row_clear_gate_2026-05-22.md
notes/vit_neural_attribute_head_training_2026-05-22.md
notes/vit_fixed_row_attribute_heads_2026-05-22.md
```

Key code now included:

```text
tools/extract_vit_candidate_features.py
tools/extract_videomae_fixed_row_features.py
tools/evaluate_vit_neural_attribute_head.py
tools/evaluate_combined_fixed_row_gate.py
tools/make_combined_fixed_row_gate_submission.py
```

This layer trains small CLIP/VideoMAE heads and a combined clear gate. It does
not persist a checkpoint; it materializes row CSVs.

### Private Component Stack

The RC1 parent submission is:

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

The OOF analog is:

```text
data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv
```

Stack provenance is documented in:

```text
notes/private_component_stack_2026-05-23.md
notes/dino_roi_attribute_smoke_2026-05-22.md
notes/dinov2roi_m03_effectiveness_candidate_2026-05-23.md
notes/effectiveness_transition_guard_worker_2026-05-23.md
notes/punch_type_component_worker_2026-05-23.md
notes/rootout_ptype_only_candidate_2026-05-23.md
notes/target_component_worker_2026-05-23.md
notes/target_guard_test_artifact_2026-05-23.md
```

Key code now included:

```text
tools/evaluate_dino_roi_attribute_model.py
tools/make_dino_roi_attribute_submission.py
tools/evaluate_effectiveness_transition_guard.py
tools/evaluate_punch_type_component_guard.py
tools/evaluate_target_component_guard.py
tools/make_target_component_guard_submission.py
tools/filter_row_updates.py
tools/combine_submission_columns.py
tools/make_submission_diff_ablation.py
```

The private stack is not one trained monolith. It is fixed-row column
composition:

- `effectiveness`: DINOv2 ROI head, guarded to landed-to-nonlanded transitions.
- `punch_type`: root-out pose/logreg transfer, guarded by explicit transition
  filters such as `transitionopt`.
- `target`: HGB target guard on miss rows with margin/stability constraints.

### Candidate-Token Clear

The RC1 final clear-drop leg trains a small Transformer at release time:

```text
tools/evaluate_candidate_token_reranker.py
tools/make_candidate_token_clear_submission.py
```

OOF p_keep cache:

```text
data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv
```

Train-all test diagnostics:

```text
data/processed/diagnostics/candidate_token_clear_trainall_test_private_stack_transitionopt_yolo26l_pose_audio_e30_top8_blockedmiss_20260523.csv
```

Documentation:

```text
notes/candidate_token_clear_strict_anchor_2026-05-23.md
notes/private_stack_clear_topk_combo_2026-05-23.md
```

No `.pt` checkpoint is expected for this model. It is seeded, trained, inferred,
and discarded inside the generator. Reproducibility is enforced by final CSV
SHA in `tools/release_build_shadow_stack_clearcut.py`.

## RC2: Old-Attribute Source Switch

Final runner now included in this branch:

```bash
.venv/bin/python tools/release_build_old_attribute_source_switch.py
```

Final outputs:

```text
data/processed/validation_rows/seq_motion_old_attr_sourcele15_093_oof.csv
submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv
```

### RC2 DAG

```text
raw videos
  -> YOLO pose tracks
  -> sequence TCN yolo26x primary + yolo11s/yolo26l witnesses
  -> exchange gate
  -> fixed-row attribute HGB
  -> fixed-row motion gate
  -> source table
  -> source switch rule old_attr.source_le15 >= 0.93
```

### Pose Weights and Tracks

The upstream sequence source uses:

```text
yolo26x-pose.pt
models/yolo11s-pose.pt
yolo26l-pose.pt
data/processed/pose_tracks/val_yolo26x_conf035/
data/processed/pose_tracks/test_yolo26x_conf035/
data/processed/pose_tracks/val_yolo11s_conf035/
data/processed/pose_tracks/test_yolo11s_conf035/
data/processed/pose_tracks/val_yolo26l_conf035/
data/processed/pose_tracks/test_yolo26l_conf035/
```

### Sequence and Postprocess Sources

Base source:

```text
data/processed/validation_rows/seq_tcn_snap4_rootcount088_repeat_exchange_attr_motion_gate_oof.csv
submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_gate_OFFLINE_CANDIDATE.csv
```

Override source:

```text
data/processed/validation_rows/hybrid_yolo26l_seq_tcn_snap4_gated_rival_exchange_attr_all_oof.csv
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_rival_exchange_attr_all_OFFLINE_CANDIDATE.csv
```

Documentation:

```text
notes/seq_repeat_exchange_attr_2026-05-21.md
notes/fighter_identity_rival_micro_current_best_2026-05-21.md
notes/exchange_state_gate_2026-05-21.md
notes/fixed_row_attribute_model_2026-05-21.md
notes/fixed_row_motion_gate_2026-05-21.md
notes/source_switch_old_attr_rule_2026-05-21.md
notes/release_old_attribute_source_switch_e2e.md
```

Key code:

```text
tools/evaluate_pose_sequence_spotter.py
tools/make_pose_sequence_submission.py
tools/evaluate_exchange_state_gate.py
tools/make_exchange_state_gate_submission.py
tools/evaluate_fixed_row_attribute_model.py
tools/make_fixed_row_attribute_submission.py
tools/evaluate_fixed_row_motion_gate.py
tools/make_fixed_row_motion_gate_submission.py
tools/build_fight_level_source_table.py
tools/evaluate_source_decision_stumps.py
tools/materialize_source_switch_rows.py
tools/release_build_old_attribute_source_switch.py
```

RC2 itself does not train a final model. It verifies and materializes a
video-level source policy over already generated row sources. The upstream row
sources do train small TCN/HGB/sklearn models, but again they materialize CSVs
rather than persistent checkpoints.

## What "Weights" Means Here

There are three categories:

1. External backbone weights: YOLO `.pt`, CLIP, VideoMAE, DINOv2. These are
   actual pretrained weights and are not committed.
2. Small local heads: TCN, Transformer, HGB/logreg/linear heads. These are
   trained during row generation and usually not saved as checkpoints.
3. Materialized row outputs: CSV/NPZ/pose-track caches under `data/processed/`
   and `submissions/`. These are generated artifacts, not source.

The branch now carries category 2 code and the command trail for category 3.
Category 1 is restored by `tools/bootstrap_release_assets.py download-models`.
Category 3 is pinned by `release_artifacts/release_artifacts_manifest.tsv` and
verified by `tools/verify_release_artifacts.py`; its rebuild trail is the notes
and scripts listed above.
