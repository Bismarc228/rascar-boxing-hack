# Release RC1 E2E - Shadow Stack ClearCut

Branch: `release/shadow-stack-clearcut`

This branch materializes the `RC1: Shadow Stack ClearCut` candidate from
`notes/release_candidates_2026-05-23.md`.

## Output

```text
submissions/prime_publicbest_nonhack_blockedonly_private_stack_transitionopt_targetguard_candidate_token_clear_yolo26l_poseaudio_e30_top8_blockedmiss_20260523_OFFLINE_CANDIDATE.csv
```

Expected SHA-256:

```text
da285e84174f8cf9489d09f7d57c8e67c1c052ec385053af47786b79de07262a
```

## Rebuild

```bash
.venv/bin/python tools/release_build_shadow_stack_clearcut.py
```

The runner performs the full release path:

1. Recomputes the strict OOF clear-drop audit on
   `component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv`.
2. Verifies the selected `pkeep_strict_current / blocked_miss / top_k=8`
   audit row has `macro_score=0.420768`, `n_kept=1180`, and `n_dropped=8`.
3. Rebuilds the train-all candidate-token clear submission from the private
   transitionopt parent.
4. Runs `tools/validate_data.py` on the generated submission.
5. Verifies the parent diff: eight `clear=true -> false` drops, split as
   `agn_037=1`, `agn_038=1`, `agn_047=6`, with `blocked=2`, `miss=6`.
6. Verifies the final submission SHA-256.

## Required Local Caches

The final CSV is ignored by Git and is rebuilt from local experiment caches:

```text
data/processed/vit_features/component_stack_dinom03_ptypetransitionopt_targetguard_strict_oof_20260523.csv
data/processed/diagnostics/candidate_token_clear_strict_current_pose_audio_yolo26l_e30_20260523.csv
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
data/processed/pose_tracks/val_yolo26l_conf035/
data/processed/pose_tracks/test_yolo26l_conf035/
submissions/prime_publicbest_nonhack_blockedonly_private_stack_dinom03_noagn037038039_rootoutptype_transitionopt_clearonly_targetguard_20260523_OFFLINE_CANDIDATE.csv
```

The legacy candidate-token reranker source is vendored in this branch as
`tools/evaluate_candidate_token_reranker.py`; the runner does not require the
old experiment branch to be checked out.
