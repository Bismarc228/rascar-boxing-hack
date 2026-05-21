# Submission Decision Matrix - 2026-05-21

Purpose: keep the upload queue explicit. This is not approval to submit.
Current guardrail remains: no Kaggle upload unless explicitly approved and the
candidate is worth spending one of the remaining attempts.

## Current Anchor

| Candidate | Evidence | Risk | Decision |
| --- | --- | --- | --- |
| `submission.csv` / `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv` | Team public best `0.16461`; validates. Latest leaderboard check at `2026-05-21T14:47:18Z` shows visible leader at `0.32088`, next visible score `0.11807`, with `8/30` submissions used today and reset at `2026-05-22T00:00:00Z`. | Already submitted; no longer overall leader. | Keep as root fallback; do not resubmit. |

## Plausible But Not Auto-Submit

| Candidate | Local evidence | Test artifact | Risk | Decision |
| --- | --- | --- | --- | --- |
| public-anchor attribute-only, effectiveness | Fixed-row attribute OOF `0.395018`, fewer test changes than all-attrs. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_OFFLINE_CANDIDATE.csv` | Attribute-only; may not move public much; still unsubmitted model output. | Safest attribute probe if an upload is explicitly allowed. |
| public-anchor attribute-only, all attrs | Fixed-row attribute OOF `0.396329`, stronger than effectiveness-only. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_OFFLINE_CANDIDATE.csv` | More attribute churn; timing/count unchanged. | Stronger but less conservative attribute probe. |
| public-anchor RGB effectiveness | RGB ViT-B/16 effectiveness replacement improves the yolo26l validation anchor `0.365177 -> 0.371229` ungated and `0.369373` at margin `0.2`; only `effectiveness` changes. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rgb_eff_m0_OFFLINE_CANDIDATE.csv`; `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_rgb_eff_m02_OFFLINE_CANDIDATE.csv`; both validate. | Attribute-only and unsubmitted, but timing/count/fighter are exactly preserved. | Superseded by stacked attr+RGB public-anchor artifacts; no automatic upload. |
| stacked public-anchor attr+RGB, effectiveness-only | Attribute effectiveness base improves `0.395018 -> 0.397369` when stacked with RGB effectiveness margin `0`; final root diff is only `effectiveness` (`345` changed ids) and clear counts are unchanged. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_rgb_eff_m0_OFFLINE_CANDIDATE.csv`; validates. | Still an unsubmitted attribute model, but it preserves timing/count/fighter/hand/target/type. | Best current fixed-row public-safe candidate if an upload is explicitly allowed. |
| stacked public-anchor attr+RGB, `agn_038` only | Same effectiveness-only stack, but spliced only onto known public-sensitive `agn_038`; validates and changes only 47 effectiveness labels with root counts unchanged. | `submissions/hybrid_root_attr_effectiveness_rgb_eff_m0_agn038_only_OFFLINE_CANDIDATE.csv`; validates. | Narrower public probe, but it may miss private lift and still changes the already public-good video. | Lowest-blast-radius public probe if an explicit single-video upload is approved. |
| stacked public-anchor attr+RGB, all attrs | Attribute-all base improves `0.396329 -> 0.398315` with RGB effectiveness margin `0.3`; strongest local public-anchor attribute score. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_rgb_eff_m03_OFFLINE_CANDIDATE.csv`; validates. | Broader attribute churn versus root: `528` changed ids across effectiveness, punch type, hand, and target. | Stronger but less conservative than effectiveness-only; no automatic upload. |
| stacked public-anchor attr+RGB, all attrs on `agn_038` only | Same strongest local `attr_all` stack, but spliced only onto `agn_038`; validates with 73 changed ids on that video. | `submissions/hybrid_root_attr_all_rgb_eff_m03_agn038_only_OFFLINE_CANDIDATE.csv`; validates. | Bounded to public-sensitive `agn_038`, but changes punch type, hand, target, and effectiveness. | Stronger single-video public probe than effectiveness-only; higher public-score risk. |
| old gated sequence private-risk | Previous gate OOF around `0.395013`; public expected mostly neutral beyond anchor. | `submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv` | Private-risk, not a public growth trigger. | Keep as older conservative private-risk baseline. |
| new sequence branch, root base, no `agn_038` | Preserves public-proven `agn_038`; replaces only `agn_047,agn_062,agn_063`; validates; `changed_vs_root=375`. | `submissions/hybrid_root_seqrepeat_exchange_attr_gate_noagn038_OFFLINE_CANDIDATE.csv` | No fresh public evidence; public may be neutral if these videos are mostly private, but this is unknown. | Most public-conservative private-risk artifact from the new branch. |
| new sequence branch, attr base, no `agn_038` | Same replace keys as above; validates; `changed_vs_root=640`. | `submissions/hybrid_attrall_seqrepeat_exchange_attr_gate_noagn038_OFFLINE_CANDIDATE.csv` | Combines two unsubmitted changes: attributes plus sequence replacements. | Less conservative than root no-`agn_038`; keep for ensemble/private. |
| new sequence branch, attr base gated | Best local splice `0.400603`; validates; replaces `agn_038,agn_047,agn_062,agn_063`; `changed_vs_root=672`. | `submissions/hybrid_attrall_seqrepeat_exchange_attr_gate_OFFLINE_CANDIDATE.csv` | Touches public-sensitive `agn_038` and expands it from `81` to `99` clear rows. | Do not upload unless explicitly accepting public-risk. |
| full new sequence branch | Strong local source `0.400288`; motion gate reaches `0.401483`; rival micro reaches `0.401575`; validates; `changed_vs_root=767` before motion/rival. | `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_all_OFFLINE_CANDIDATE.csv`; `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_rival_micro_OFFLINE_CANDIDATE.csv` | Full-sequence public transfer already failed in prior probes. | Do not submit directly. |
| motion-gated new sequence branch, root base, no `agn_038` | Best local source branch is motion-gated (`0.401483`), but this conservative test splice preserves public-proven `agn_038`; validates; `changed_vs_root=375`. | `submissions/hybrid_root_seqrepeat_exchange_attr_motion_gate_noagn038_OFFLINE_CANDIDATE.csv` | OOF lift is small, and private-risk videos still have no fresh public evidence. | Most conservative artifact from the current best local branch. |
| motion-gated new sequence branch + rival micro, root base, no `agn_038` | Narrow opposite-fighter rival rule on the current best OOF gives only `0.401483 -> 0.401575` with 20 changed OOF rows. The conservative test splice validates and changes 13 clear rows versus its motion-gated parent. | `submissions/hybrid_root_seqrepeat_exchange_attr_motion_rival_micro_noagn038_OFFLINE_CANDIDATE.csv` | Increment is tiny and inherits sequence public-transfer risk. | Save for ensemble only; not worth an upload. |
| old-attribute source switch by `source_le15>=0.93` | Validation splice reaches `0.410780` over current best `0.401483`, but the lift is dominated by `agn_003`; excluding `agn_003` regresses `0.419714 -> 0.418878`, and group-OOF stumps score only `0.399638`. | `submissions/seq_motion_old_attr_sourcele15_093_OFFLINE_CANDIDATE.csv` | Direct test analog selects `agn_038,agn_062,agn_063`; `agn_038` is public-sensitive, and the rule is fitted on only 13 validation videos. | Save as ensemble/source-policy hypothesis; do not upload automatically. |
| fixed-row audio gate on full current source | OOF improves current best `0.401483 -> 0.404479`; audio-only is neutral and pose-only is only `0.401999`, so this is a real auxiliary signal. | `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_gate_OFFLINE_CANDIDATE.csv` | Drops 29 clear test rows from full sequence source, including `agn_038`; still inherits sequence public-transfer risk. | Keep for ensemble/private-risk queue; no automatic upload. |
| fixed-row audio gate on root no-`agn_038` splice | Same audio gate applied to the conservative root splice with `agn_038` protected; validates and drops 21 clear rows (`733 -> 712`). | `submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_gate_noagn038_OFFLINE_CANDIDATE.csv` | Public-sensitive `agn_038` preserved, but this still changes private-risk sequence splice videos and has no direct public evidence. | Safer audio-gate artifact for future explicitly approved private-risk upload; do not auto-submit. |
| audio-gate exchange-side identity micro | OOF gain is tiny (`0.404479 -> 0.404915`); generated full and protected test artifacts both validate and change only 11 fighter labels. | `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_OFFLINE_CANDIDATE.csv`; `submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_noagn038_OFFLINE_CANDIDATE.csv` | Too small to justify an upload alone; useful only as a future ensemble/private-risk micro-signal. | Save; no automatic upload. |
| tracklet-appearance fighter identity micro | OOF improves the audio-gate exchange-side source only `0.404915 -> 0.405200`, changing one validation fighter label. | No test CSV generated. | Useful as an ensemble diagnostic, but one-row validation gain is not upload evidence. | Save; no automatic upload and do not generate a Kaggle artifact unless it is part of a larger approved ensemble. |
| RGB fixed-row effectiveness micro | OOF improves the current best source `0.405200 -> 0.407232` with confidence margin `0.2`, changing only `effectiveness` on 367 validation rows. On the nearest test-capable source it improves `0.404915 -> 0.406939`. | Full and protected audio-exchange RGB artifacts validate: `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_audio_exchange_side_rgb_eff_m02_OFFLINE_CANDIDATE.csv`, `submissions/hybrid_root_seqrepeat_exchange_attr_motion_audio_exchange_side_rgb_eff_m02_noagn038_OFFLINE_CANDIDATE.csv`. | Attribute-only and still inherits sequence public-risk. | Save as local best OOF/test-capable micro-source; no automatic upload. |
| strict public-neutral RGB/audio-exchange splice | Uses the latest RGB/audio-exchange source only on `agn_047,agn_062,agn_063`; preserves public-good `agn_038` and sequence-bad/public-risk `agn_037`; validates; `changed_vs_root=375`. | `submissions/hybrid_root_audio_exchange_rgb_eff_m02_publicneutral_047_062_063_OFFLINE_CANDIDATE.csv` | Expected public-neutral because the same videos were individually neutral before; not a reason to beat `0.16461`, let alone the new visible leader at `0.32088`. | Safest current private-risk artifact, but still no automatic upload. |

## Rejected For Upload

| Candidate | Reason |
| --- | --- |
| row-level source stacker | OOF `0.368373`, below current sources. |
| yolo26l learned temporal selector | OOF `0.275094`, below heuristic. |
| full old sequence TCN | Public `0.12712`, already failed. |
| full yolo26x/yolo11s agreement | Public `0.10784`, already failed. |
| crop-motion direct submit | Local gain is weak and source oracle contribution is marginal. |
| temporal frozen RGB contact fixed-row branch | Union and attacker/opponent ViT-B clip smokes both regressed below the current best OOF source; no test CSV generated. |
| candidate-level RGB contact raw-pose branch | Improves same-pool raw pose OOF, but best smoke score `0.359641` is far below current best `0.407232`; no test CSV generated. |
| sequence-TCN + RGB contact bridge | Full pool-1800 A/B confirms RGB helps the sequence scorer (`0.386043 -> 0.391237`), but the final source is still below current best `0.407232`; no test CSV generated. |
| sequence-TCN + audio contact bridge cap-400 | Medium OOF `0.379792`, below current best `0.407232`; no test CSV generated. |
| deep crop per-video fighter identity calibration | ResNet50 crop remapping regressed even with oracle cluster mapping (`0.361655 -> 0.342864`); no test CSV generated. |
| tracklet-appearance fighter identity alone | Positive but only micro-sized (`+0.000285` over current best, one OOF flip); no test CSV generated. |
| learned exchange-side fighter flip | Best OOF `0.401720` is only `+0.000236` over current best; no test CSV generated. |
| automatic row-source policy with RGB/audio/tracklet sources | Oracle headroom is high (`0.430014`), but current policies stay below the best single source (`hgb=0.404206` vs `0.407232`); no test CSV generated. |
| stacked-attribute source policy | Focused oracle with stacked attribute sources reaches `0.422839`, and a preserve-order ridge policy materializes `0.407417`, but the lift over `rgb_eff_m02` is only `+0.000185`, stumps regress to `0.392902`, and there is no clean test analog. |
| public-anchor source policy over stacked attribute sources | Testable source-policy audit collapses to existing artifacts: mean/stump choose full `attr_all_rgb`, ridge is slightly below it (`0.398149` vs `0.398315`), and HGB regresses. No new CSV generated. |

## Current Recommendation

Do not upload automatically. If an upload is explicitly approved, the least bad
order is:

1. Stacked public-anchor `attr_effectiveness_rgb_eff_m0` if the goal is fixed
   timing/count/fighter and only `effectiveness` changes. Use the `agn_038`-only
   splice for the lowest-blast-radius public probe; use the full artifact for a
   broader public/private attribute probe. `attr_all_rgb_eff_m03`, full or
   `agn_038`-only, is stronger locally but less conservative.
2. `hybrid_root_audio_exchange_rgb_eff_m02_publicneutral_047_062_063_OFFLINE_CANDIDATE.csv`
   if the goal is the safest current private-risk splice with `agn_038` and
   `agn_037` preserved.
3. Avoid full sequence or `agn_038`-expanded variants unless the user explicitly
   accepts public-score risk.
