# Punch Type Component Worker - 2026-05-23

Scope: revived hypothesis #5, fixed rows only, `punch_type` only. No Kaggle
upload.

## Inputs

Strict OOF anchor:

```text
data/processed/vit_features/clip_ptype_m028_videomae_neural_eff_fighter_target_clear_ensemble_posehgb094_vitlogreg012_rows_20260522.csv
```

Compared on strict anchor rows:

```text
data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m02_rows_20260522.csv
data/processed/vit_features/clip_vitb16_attackerdef_glovetarget_t3_current_oof_cuda_preproc_ptype_m028_rows_20260522.csv
data/processed/validation_rows/candidate_token_punch_type_margin095_20260522_oof.csv
data/processed/diagnostics/candidate_token_punch_type_pose_audio_rgb_e40_20260522.csv
```

Root-out pose-logreg source:

```text
data/processed/root_out/yolo11s_pose_geom_id_attr_model_pose_logreg_noctx_c100_joint_margin_20260521/
variant=punch_type_margin_0.1
```

For root-out, the source-before-logreg rows were reconstructed from
`replacement_audit.csv` source columns. This reproduces the summary baseline
`macro=0.294997593`, `score_punch_type=0.175548891`.

## Commands

```bash
python3 -m py_compile tools/evaluate_punch_type_component_guard.py
python3 tools/evaluate_punch_type_component_guard.py
git diff --check
```

The guard checks row count and all non-`punch_type` submission columns before
writing outputs.

## Summary

Pass rule: `score_punch_type >= 0.235` or macro delta `>= +0.003`, jab and
uppercut recall improve, and no per-root punch_type delta below `-0.01`.

```text
split    variant                                      macro       d_macro    ptype      d_ptype    jab_d      uppercut_d  min_root_d  changed  pass
strict   clip_margin_m028_anchor                      0.415261   +0.000000  0.211716  +0.000000  +0.000000  +0.000000   +0.000000       0    baseline
strict   clip_margin_m02_on_anchor                    0.415204   -0.000057  0.211142  -0.000574  +0.008287  +0.010753   -0.001349      64    fail
strict   candidate_token_margin095_source_on_anchor   0.415162   -0.000099  0.210723  -0.000993  -0.074586  -0.069892   -0.049590     509    fail
strict   candidate_token_margin_0p975_on_anchor       0.416064   +0.000803  0.219747  +0.008031  -0.008287  -0.021505   -0.033489     268    fail
strict   pose_logreg_m01_frame_fighter_hand_on_anchor 0.415837   +0.000576  0.217477  +0.005761  -0.016575  -0.005376   -0.013645     130    fail
rootout  source_before_pose_logreg                    0.294998   +0.000000  0.175549  +0.000000  +0.000000  +0.000000   +0.000000       0    baseline
rootout  pose_logreg_punch_type_margin_0p1            0.299664   +0.004666  0.222214  +0.046665  +0.126635  +0.167722   +0.028773    2821    pass
```

Strict candidate-token has a small best macro lift at margin `0.975`, but it
fails the recall guard and regresses the `бокс` root by `-0.033489` ptype. The
strict pose-logreg aligned transfer is only partial (`942` anchor rows had no
exact `video_key,frame,fighter,hand` source match) and also fails recall/root
guards.

## Confusion

Counts are GT punch type rows by predicted punch type; `missed` includes
unmatched or unscorable rows.

```text
split    variant                                 gt        jab  cross  hook  uppercut  missed  total  recall
strict   clip_margin_m028_anchor                 jab        71     49    55        11     176    362   0.196133
strict   clip_margin_m028_anchor                 cross     113    122   123        17     163    538   0.226766
strict   clip_margin_m028_anchor                 hook       68     72   189        50     277    656   0.288110
strict   clip_margin_m028_anchor                 uppercut   16     12    56        20      82    186   0.107527
strict   candidate_token_margin_0p975_on_anchor  jab        68     58    52         8     176    362   0.187845
strict   candidate_token_margin_0p975_on_anchor  cross      97    148   121         9     163    538   0.275093
strict   candidate_token_margin_0p975_on_anchor  hook       68     66   205        40     277    656   0.312500
strict   candidate_token_margin_0p975_on_anchor  uppercut   14     14    60        16      82    186   0.086022
rootout  source_before_pose_logreg               jab         0    264   382         0     807   1453   0.000000
rootout  source_before_pose_logreg               cross       0    778   904         0    1057   2739   0.284045
rootout  source_before_pose_logreg               hook        0    486  1133         0    1510   3129   0.362097
rootout  source_before_pose_logreg               uppercut    0     86   206         0     340    632   0.000000
rootout  pose_logreg_punch_type_margin_0p1       jab       184    248   122        92     807   1453   0.126635
rootout  pose_logreg_punch_type_margin_0p1       cross     291    856   338       197    1057   2739   0.312523
rootout  pose_logreg_punch_type_margin_0p1       hook      220    362   756       281    1510   3129   0.241611
rootout  pose_logreg_punch_type_margin_0p1       uppercut   28     54   104       106     340    632   0.167722
```

Root-out pose-logreg is the only revived #5 component that passes this guard:
it creates non-zero jab and uppercut recall from a source that had none, while
keeping rows and all other attributes fixed.

## Per Root

```text
split    variant                                root             macro      ptype      ptype_delta
strict   candidate_token_margin_0p975_on_anchor Турнир Бокс      0.424229   0.245637  +0.015763
strict   candidate_token_margin_0p975_on_anchor Турнир Бокс 2    0.446792   0.220216  +0.022512
strict   candidate_token_margin_0p975_on_anchor бокс             0.336202   0.192763  -0.033489
strict   pose_logreg_m01_frame_fighter_hand_on_anchor Турнир Бокс 0.423678  0.240121  +0.010247
strict   pose_logreg_m01_frame_fighter_hand_on_anchor Турнир Бокс 2 0.445756 0.209860  +0.012156
strict   pose_logreg_m01_frame_fighter_hand_on_anchor бокс        0.338186  0.212607  -0.013645
rootout  pose_logreg_punch_type_margin_0p1      Турнир Бокс      0.309113   0.234205  +0.057827
rootout  pose_logreg_punch_type_margin_0p1      Турнир Бокс 2    0.318224   0.244092  +0.043389
rootout  pose_logreg_punch_type_margin_0p1      бокс             0.250059   0.161968  +0.028773
```

## Artifacts

```text
data/processed/diagnostics/punch_type_component_summary_20260523.csv
data/processed/diagnostics/punch_type_component_confusion_20260523.csv
data/processed/diagnostics/punch_type_component_per_root_20260523.csv
data/processed/diagnostics/punch_type_component_changes_20260523.csv
data/processed/vit_features/punch_type_component_strict_token_best_rows_20260523.csv
data/processed/vit_features/punch_type_component_strict_pose_logreg_aligned_rows_20260523.csv
data/processed/vit_features/punch_type_component_rootout_pose_logreg_m01_rows_20260523.csv
```
