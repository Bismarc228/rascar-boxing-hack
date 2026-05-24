# MR note: rebuild clearens test artifact

Target branch: `research/new-growth-opportunities`

Source branch: `experiments/rebuild-clearens-20260522`

## Summary

Restores the combined fixed-row clear-gate toolchain from the ViT detector
branch and uses it to recover the missing test-side stricter clearens artifact.

## Evidence

The recovered toolchain exactly reproduces the current submitted non-hack
clearens mask:

```text
input:
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_t07_target_m090_20260522.csv

reproduction:
submissions/debug_rebuild_clearens090_008_directtrain_from_tools_20260522.csv

current anchor:
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens090_008_20260522.csv

changed_ids=0
clear_rows=706
```

The stricter recovered candidate is:

```text
submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_20260522_OFFLINE_CANDIDATE.csv
```

It validates and has:

```text
clear_rows=692
changed_vs_current_anchor=14 clear-only true->false rows
```

## Risk

This is not an upload recommendation by itself. The corresponding OOF stricter
gate was locally positive, but the test artifact drops 14 additional rows
versus the current public anchor, including 2 landed rows and one `agn_038`
row. Treat it as a private-risk candidate that requires explicit upload
approval and a current public/private-risk decision.

## Validation

```bash
python3 -m py_compile tools/evaluate_vit_fixed_row_attribute_head.py tools/evaluate_vit_neural_attribute_head.py tools/evaluate_vit_fixed_row_gate.py tools/evaluate_combined_fixed_row_gate.py tools/make_combined_fixed_row_gate_submission.py
python3 tools/validate_data.py --submission submissions/debug_rebuild_clearens090_008_directtrain_from_tools_20260522.csv
python3 tools/validate_data.py --submission submissions/prime_publicbest_nonhack_clipptype_m028_eff_d2res_focal_m010_fighter_target_clearens094_012_20260522_OFFLINE_CANDIDATE.csv
git diff --check
```

No Kaggle upload was made.
