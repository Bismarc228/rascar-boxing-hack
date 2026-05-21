# Public LB Strategy

Context as of `2026-05-21T00:20Z`: quota reset happened and `8/30` submissions
have been used today. User-imposed budget is now stricter than Kaggle quota:
**at most 2 more submissions today, and only for a clear breakthrough**.

Current public best:

```text
0.16461  submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv
```

The root-level `submission.csv` is currently copied from this public-best file
and passes `tools/validate_data.py`.

Leaderboard check at `2026-05-21T00:40Z`: team `Los Pollos Hermanos` is listed
with score `0.16461`; the next visible public score is `0.04481`. This supports
holding the remaining attempts instead of probing blindly.

This is a hybrid of the old public anchor plus sequence-TCN snap4 predictions
only for `agn_038`. It beats the previous yolo26l public anchor:

```text
0.13849  submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
```

## Post-Reset Public Results

```text
0.16461  hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088
0.16461  hybrid_yolo26l_best_agn038_agn047_seq_tcn_snap4_rootcount088
0.16461  hybrid_yolo26l_best_agn038_agn062_seq_tcn_snap4_rootcount088
0.16461  hybrid_yolo26l_best_agn038_agn063_seq_tcn_snap4_rootcount088
0.12888  hybrid_yolo26l_best_agn038_yolo26x_yolo11s_agree
0.12712  seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount088
0.12712  hybrid_yolo26l_best_agn038_agn037_seq_tcn_snap4_rootcount088
0.10784  yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088
```

Readout:

- Sequence-TCN snap4 is strong on `agn_038`; it lifted public from `0.13849`
  to `0.16461` when applied only there.
- Full sequence-TCN is not public-safe. It dropped to `0.12712`.
- yolo26x+yolo11s agreement is not public-safe. Full submit scored `0.10784`,
  and even `agn_038`-only hybrid scored only `0.12888`.
- Adding sequence replacements for `agn_047`, `agn_062`, or `agn_063` did not
  move public from `0.16461`; these are likely private or neutral for public.
- Adding sequence replacement for `agn_037` dropped to `0.12712` even with the
  same selected count (`52`). Local CSV comparison shows a large timing
  distribution shift on `agn_037`, so treat it as public and sequence-bad.

## Stop Rule

No more uploads by default. The remaining Kaggle quota is not the operative
limit; the operative limit is the user's cap of at most two more attempts.

A new submit must satisfy all of:

- It is not a full yolo26x agreement or full sequence-TCN replacement.
- It preserves the known-good `agn_038` sequence snap4 replacement.
- It does not alter `agn_037` unless a local/video-specific fix explains the
  timing failure.
- It has a concrete reason to beat `0.16461`, not just another blind hybrid.
- The CSV passes `tools/validate_data.py`.

## Next Work Without Submits

1. Audit `agn_038` sequence rows versus yolo26l base to understand why it wins:
   count, frame snapping, fighter label, or attribute changes.
2. Audit `agn_037` sequence rows versus yolo26l base to isolate the timing
   collapse. Count is unchanged, so compare frame distribution and candidate
   source confidence.
3. Test `agn_048`, `agn_039`, `agn_049`, and `agn_064` locally by diffing frame
   movement before considering a public hybrid. These videos have equal counts
   between yolo26l base and sequence, so timing/identity changes dominate.
4. Build a safer hybrid generator that only replaces a video if sequence and
   yolo26l agree on frame neighborhoods, or if the sequence replacement matches
   the `agn_038` improvement pattern.
5. Keep full sequence and yolo26x agreement as offline research artifacts, not
   submission candidates.

Implemented helper: `tools/gate_submission_hybrid.py`. With default gates
(`count_delta<=20`, base/override `<=15f` overlap at least `0.80`, median gap
at most `15f`), it selects only:

```text
agn_038,agn_047,agn_062,agn_063
```

It rejects:

```text
agn_037,agn_039,agn_048,agn_049,agn_064
```

The generated local candidate
`submissions/hybrid_yolo26l_best_seq_tcn_snap4_gated_local_OFFLINE_CANDIDATE.csv`
passes validation, but is not a submit candidate yet. Public already showed
`agn_047/062/063` are neutral when added individually, so this is a possible
private-risk candidate, not a public breakthrough.

The helper now has two output modes: `--row-mode sample` for fixed Kaggle
submission rows and `--row-mode video_rows` for validation prediction rows.
Both modes were smoke-tested on the `agn_038` sequence replacement.

Sequence validation-row export is now wired in
`tools/evaluate_pose_sequence_spotter.py` via `--write-best-rows`; a short
GPU-1 smoke run confirmed it writes rows. The next non-submit validation step is
to run a proper sequence OOF export, generate matching yolo26l-base validation
rows, and score the gated hybrid locally.

That validation step was run with the sequence snap4 `root_count=0.88` setup:

```text
0.365177  yolo26l same_sum w4 alpha=-0.2 root_rate=0.88 validation rows
0.379682  full sequence snap4 root_count=0.88 OOF rows
0.389125  gated yolo26l/sequence hybrid OOF rows
```

The gate selected tournament videos with local frame agreement and rejected the
old `бокс` videos plus `agn_072`, where sequence had high count/FP risk. This
supports the gate as a real validation improvement, not just a public-probe
patch. It still does not justify another public upload by itself because the
matching test gated candidate is expected to keep the same public score
(`0.16461`) while changing likely-private videos.

A threshold sweep with `tools/evaluate_gate_hybrid_rows.py` improved the OOF
gate further:

```text
0.395013  max_count_delta=48, min_base15=0.85,
          min_override15=0.70, max_median_gap=4..30
```

Applied to test, this gate selects `agn_038,agn_062,agn_063` and rejects the
known-bad `agn_037`. The matching CSV is validated:

```text
submissions/hybrid_yolo26l_best_seq_tcn_snap4_gate_oof395_test_OFFLINE_CANDIDATE.csv
selected total=731
```

Do not upload this yet. It is a stronger private-risk candidate, but public is
unlikely to rise because `agn_062/063` were individually public-neutral.

Per-video OOF oracle for choosing base vs sequence reaches `0.401371`. The gate
captures most of that (`0.395013`), but still makes validation mistakes:
sequence is bad on `agn_010` and `agn_056`, while useful on videos such as
`agn_004` and `agn_072`. Nearest-frame/count thresholds alone cannot separate
all of these cases; the next improvement needs richer per-video features or a
fight/root-aware policy, not another blind threshold tweak.

`tools/analyze_gate_oracle_features.py` now prints this per-video table. The
most relevant miss for test is `agn_056` (`Турнир Бокс 2`, fight 5, round 1):
it passes local overlap/count gates but sequence is worse. That weakens the
case for submitting the `agn_062/063` private-risk replacement without another
signal.

`tools/evaluate_gate_policy_model.py` tested a small Ridge per-video policy on
the same features. It was worse than the threshold gate (`0.381635` fight-CV and
`0.380425` video-CV versus `0.395013`), so learned per-video policy is not a
submit path with only the current 13 validation videos.

`root_count=0.92` and `root_count=0.82` sequence variants were checked through
the same gate. They select essentially the same safe video set. On `agn_038`,
`root_count=0.88` and `0.92` have `98.8%` of selected frames within 3 frames of
each other; `0.92` mainly adds a couple of extra far events. This is not enough
evidence to spend one of the two remaining submits.

The same gate also rejects yolo26x agreement on `agn_038` because it adds too
many rows there (`+34` for yolo26x+yolo11s, `+30` for yolo26x+yolo26l). This
matches the public failure of the `agn_038` agreement hybrid (`0.12888`), so
there is no gated yolo26x public candidate right now.

## Local Base-vs-Sequence Diff

Rough nearest-frame comparison between yolo26l public base and sequence snap4
`root_count=0.88` explains why broad replacement is unsafe:

```text
video    count_delta  base events with seq frame <=15/30/90f  median nearest gap
agn_037  +0           0.42 / 0.44 / 0.46                    358.5f
agn_038  +7           0.85 / 0.88 / 0.93                      1.0f
agn_039  +0           0.70 / 0.70 / 0.70                      1.0f, high tail
agn_047 +13           0.83 / 0.87 / 0.96                      1.0f
agn_048  +0           0.39 / 0.39 / 0.41                    336.5f
agn_049  +0           0.42 / 0.42 / 0.44                    236.5f
agn_062 +12           0.88 / 0.93 / 0.99                      1.0f
agn_063 +16           0.93 / 0.96 / 1.00                      1.0f
agn_064  +0           0.64 / 0.65 / 0.68                      2.0f, high tail
```

Interpretation:

- `agn_038` has mostly local frame changes and is public-good.
- `agn_037`, `agn_048`, and `agn_049` are high-risk timing rewrites despite
  unchanged counts.
- `agn_039` and `agn_064` have near matches for many rows but large tails; not
  enough evidence for a submit under the two-attempt cap.
- `agn_047`, `agn_062`, and `agn_063` look locally consistent and were
  public-neutral; they may be private, so changing them is a private-risk
  decision, not a public-score play.

## Operational Guardrails

- GPU 0 stays unused. Any future GPU job must use physical GPU 1:
  `CUDA_VISIBLE_DEVICES=1` or the repo's `--cuda-visible-devices 1` wrapper.
- Verify long GPU jobs with:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
- Use 10-second polling waits for Kaggle status checks.
- Every CSV must pass `tools/validate_data.py` before upload.
