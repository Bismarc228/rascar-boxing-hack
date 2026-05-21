# Public LB Strategy

Context as of `2026-05-21T00:20Z`: quota reset happened and `8/30` submissions
have been used today. User-imposed budget is now stricter than Kaggle quota:
**at most 2 more submissions today, and only for a clear breakthrough**.

Current public best:

```text
0.16461  submissions/hybrid_yolo26l_best_agn038_seq_tcn_snap4_rootcount088_OFFLINE_CANDIDATE.csv
```

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

## Operational Guardrails

- GPU 0 stays unused. Any future GPU job must use physical GPU 1:
  `CUDA_VISIBLE_DEVICES=1` or the repo's `--cuda-visible-devices 1` wrapper.
- Verify long GPU jobs with:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
- Use 10-second polling waits for Kaggle status checks.
- Every CSV must pass `tools/validate_data.py` before upload.
