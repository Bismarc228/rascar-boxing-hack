# Ensemble Candidate Registry

Purpose: keep only independent, non-dead branches that may be useful for a
future ensemble. This is not a submit queue. A row can be kept here with weak
public transfer if it provides a different signal for timing, count, identity,
or gating.

Current hard guardrail: no default Kaggle uploads. At most two more submissions
today, only if offline evidence clearly beats the current public anchor.

## Current Anchor

| Branch | Evidence | Artifact | Ensemble role | Submit status |
| --- | --- | --- | --- | --- |
| yolo26l public base + sequence snap4 only on `agn_038` | Public `0.16461`, current best. Keeps public-proven base and edits only the one confirmed good sequence video. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv`; root `submission.csv` points here. | Main anchor / fallback. | Already submitted; do not resubmit. |

## Keep For Ensemble

| Branch | Evidence | Artifact | Ensemble role | Submit status |
| --- | --- | --- | --- | --- |
| yolo26l same-sum public base | OOF validation rows `0.365177`; public `0.13849`. Precision-oriented count control transferred better than denser local winners. | `data/processed/validation_rows/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088.csv`; `submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv`. | Conservative timing/count base. | Already submitted; not enough alone now. |
| sequence TCN snap4 rootcount088 | Full OOF `0.379682`, better than yolo26l validation, but full public submit was only `0.12712`. Public failure is video-local, not total death. | `data/processed/validation_rows/seq_tcn_snap4_rootcount088_oof.csv`; `submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount088_OFFLINE_CANDIDATE.csv`. | Timing witness and replacement pool, never direct full replacement. | Already submitted full and failed; keep only as per-video source. |
| yolo26l/sequence threshold-gated hybrid | OOF gate sweep reached `0.395013`; matching test gate selects `agn_038,agn_062,agn_063` and rejects `agn_037`. Expected public-neutral beyond anchor but plausible private lift. | `submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv`. | Private-risk per-video replacement set. | Not submitted under current cap. |
| crop-motion rescored yolo26l context | Full validation `0.365177 -> 0.368244`; improves time/fighter but increases FP and rewrites many selected events. Test artifact has `707` clear rows, `+11` versus the yolo26l base, with `420` rowwise frame/fighter/hand/target/clear differences. Pairing with sequence source only raises source oracle `0.401371 -> 0.401757`, so this is marginal. | Validation rows: `data/processed/validation_rows/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008.csv`. Test CSV: `submissions/yolo26l_cropmotion_samesum_w4_am02_thr085_same10_cross4_rootrate088_ma-004_mb008_OFFLINE_CANDIDATE.csv`. Repro config: `tools/make_crop_motion_context_submission.py --tracks-dir data/processed/pose_tracks/test_yolo26l_conf035 --threshold 0.85 --nms-frames 10 --cross-nms-frames 4 --count-mode root_rate --count-multiplier 0.88 --context-feature same_sum --context-window 4 --context-alpha -0.2 --resize-width 320 --motion-alpha -0.04 --motion-beta 0.08 --jobs 16 --quiet`. | Weak independent timing tie-breaker; only useful inside ensemble/vote, not as direct submit. | Validated locally; do not submit alone. |
| opposite-fighter rival flip | Fixed-row identity audit showed role/color flips are dead, but a narrow nearby opposite-fighter pose rival rule is weakly positive across sources: yolo26l `0.365177 -> 0.365447`, sequence `0.379682 -> 0.380125`, gated hybrid `0.389125 -> 0.389672`. Full source oracle with base/seq/motion/rival reaches `0.402103`, with rival uniquely winning `agn_024`, `agn_025`, `agn_070`. | Audit: `notes/fighter_identity_error_audit_2026-05-21.md`. Test CSV: `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rival_w0_samehand_r13_min1_fht_OFFLINE_CANDIDATE.csv`, validated with `tools/validate_data.py`. | High-precision identity micro-signal for future ensemble, not a standalone branch. | Not submitted; gain is too small under current guardrail. |

## Watchlist, Not Direct Candidates

| Branch | Reason to keep | Kill condition |
| --- | --- | --- |
| yolo26x/yolo11s agreement | Good OOF before public, but public full submit was `0.10784`; may still be useful as an agreement/witness feature around selected events. | Do not use as row source unless a video-local gate proves it. |
| audio onset features | Audio-only and hard snap are killed, but local onset max/distance can remain as weak features for a learned pose-dominant selector. | Do not generate audio-only rows or hard-shift selected frames to audio peaks. |
| fighter identity visual embeddings | Existing HSV/bbox clustering and simple role/color flips failed; robust embeddings from pose-guided crops are still untested and independent from timing. | Keep only if fixed-timing validation improves fighter score without hurting time/count. |
