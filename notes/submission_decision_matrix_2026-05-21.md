# Submission Decision Matrix - 2026-05-21

Purpose: keep the upload queue explicit. This is not approval to submit.
Current guardrail remains: no Kaggle upload unless explicitly approved and the
candidate is worth spending one of the remaining attempts.

## Current Anchor

| Candidate | Evidence | Risk | Decision |
| --- | --- | --- | --- |
| `submission.csv` / `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv` | Public best `0.16461`; validates. | Already submitted. | Keep as root fallback; do not resubmit. |

## Plausible But Not Auto-Submit

| Candidate | Local evidence | Test artifact | Risk | Decision |
| --- | --- | --- | --- | --- |
| public-anchor attribute-only, effectiveness | Fixed-row attribute OOF `0.395018`, fewer test changes than all-attrs. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_effectiveness_OFFLINE_CANDIDATE.csv` | Attribute-only; may not move public much; still unsubmitted model output. | Safest attribute probe if an upload is explicitly allowed. |
| public-anchor attribute-only, all attrs | Fixed-row attribute OOF `0.396329`, stronger than effectiveness-only. | `submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_attr_all_OFFLINE_CANDIDATE.csv` | More attribute churn; timing/count unchanged. | Stronger but less conservative attribute probe. |
| old gated sequence private-risk | Previous gate OOF around `0.395013`; public expected mostly neutral beyond anchor. | `submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv` | Private-risk, not a public growth trigger. | Keep as older conservative private-risk baseline. |
| new sequence branch, root base, no `agn_038` | Preserves public-proven `agn_038`; replaces only `agn_047,agn_062,agn_063`; validates; `changed_vs_root=375`. | `submissions/hybrid_root_seqrepeat_exchange_attr_gate_noagn038_OFFLINE_CANDIDATE.csv` | No fresh public evidence; public may be neutral if these videos are mostly private, but this is unknown. | Most public-conservative private-risk artifact from the new branch. |
| new sequence branch, attr base, no `agn_038` | Same replace keys as above; validates; `changed_vs_root=640`. | `submissions/hybrid_attrall_seqrepeat_exchange_attr_gate_noagn038_OFFLINE_CANDIDATE.csv` | Combines two unsubmitted changes: attributes plus sequence replacements. | Less conservative than root no-`agn_038`; keep for ensemble/private. |
| new sequence branch, attr base gated | Best local splice `0.400603`; validates; replaces `agn_038,agn_047,agn_062,agn_063`; `changed_vs_root=672`. | `submissions/hybrid_attrall_seqrepeat_exchange_attr_gate_OFFLINE_CANDIDATE.csv` | Touches public-sensitive `agn_038` and expands it from `81` to `99` clear rows. | Do not upload unless explicitly accepting public-risk. |
| full new sequence branch | Strong local source `0.400288`; motion gate reaches `0.401483`; rival micro reaches `0.401575`; validates; `changed_vs_root=767` before motion/rival. | `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_all_OFFLINE_CANDIDATE.csv`; `submissions/seq_tcn_yolo26x_witness_repeat_thr06_nms10_cross2_snap4_rootcount088_exchange_attr_motion_rival_micro_OFFLINE_CANDIDATE.csv` | Full-sequence public transfer already failed in prior probes. | Do not submit directly. |
| motion-gated new sequence branch, root base, no `agn_038` | Best local source branch is motion-gated (`0.401483`), but this conservative test splice preserves public-proven `agn_038`; validates; `changed_vs_root=375`. | `submissions/hybrid_root_seqrepeat_exchange_attr_motion_gate_noagn038_OFFLINE_CANDIDATE.csv` | OOF lift is small, and private-risk videos still have no fresh public evidence. | Most conservative artifact from the current best local branch. |
| motion-gated new sequence branch + rival micro, root base, no `agn_038` | Narrow opposite-fighter rival rule on the current best OOF gives only `0.401483 -> 0.401575` with 20 changed OOF rows. The conservative test splice validates and changes 13 clear rows versus its motion-gated parent. | `submissions/hybrid_root_seqrepeat_exchange_attr_motion_rival_micro_noagn038_OFFLINE_CANDIDATE.csv` | Increment is tiny and inherits sequence public-transfer risk. | Save for ensemble only; not worth an upload. |

## Rejected For Upload

| Candidate | Reason |
| --- | --- |
| row-level source stacker | OOF `0.368373`, below current sources. |
| yolo26l learned temporal selector | OOF `0.275094`, below heuristic. |
| full old sequence TCN | Public `0.12712`, already failed. |
| full yolo26x/yolo11s agreement | Public `0.10784`, already failed. |
| crop-motion direct submit | Local gain is weak and source oracle contribution is marginal. |

## Current Recommendation

Do not upload automatically. If an upload is explicitly approved, the least bad
order is:

1. Public-anchor `effectiveness` or `all_attrs` attribute-only probe if the goal
   is public safety.
2. `hybrid_root_seqrepeat_exchange_attr_motion_gate_noagn038_OFFLINE_CANDIDATE.csv` if
   the goal is private-risk with public-sensitive `agn_038` preserved.
3. Avoid full sequence or `agn_038`-expanded variants unless the user explicitly
   accepts public-score risk.
