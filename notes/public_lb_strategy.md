# Public LB Strategy

Context as of `2026-05-20T23:57:51Z`: daily quota is still exhausted
(`30/30`) and the Kaggle helper reports reset at `2026-05-21T00:00:00Z`.
Do not submit before reset. Public leaderboard is roughly `66%` of test by
insider signal, so public movement is useful, but not a complete proxy for the
private `34%`.

Current public best:

```text
0.13849  submissions/yolo26l_samesum_w4_am02_thr085_same10_cross4_rootrate088_OFFLINE_CANDIDATE.csv
```

The public probes showed that at least `agn_038` affects public score. Zeroing
`agn_038` dropped the public score to `0.06940`; zeroing `agn_039`, `agn_047`,
`agn_049`, `agn_062`, `agn_063`, or `agn_064` did not move the checked score
for that probe baseline. Do not spend more quota on mask probes until the main
candidate queue is tested.

## Post-Reset Submit Queue

Submit only after quota is fresh and only one candidate at a time, with
`submissions` checked after each upload.

1. First submit:
   `submissions/yolo26x_yolo11s_agree_w4_a02_pw10_sw08_thr14_nms10_cross2_rootcount088_OFFLINE_CANDIDATE.csv`
   - Offline `0.377949`, time `0.533478`, FP `0.086935`, wins `10/13`.
   - Root audit has no risk flags: `Турнир Бокс +0.032473`,
     `Турнир Бокс 2 +0.038638`, `бокс +0.037968`.
   - Test selected rows: `727`.
   - Reason: safest materially new candidate, lower FP than yolo26x+yolo26l
     agreement, and less count-risky than direct threshold-count yolo26x.

2. If the first submit does not regress badly, submit:
   `submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount088_OFFLINE_CANDIDATE.csv`
   - First sweep offline `0.389963`, repeat audit `0.387039`.
   - Test selected rows: `744`.
   - Reason: strongest learned spotter with lower test count than the older
     sequence `root_count=0.92` candidate.

3. If the safer sequence TCN transfers, submit:
   `submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross2_snap4_rootcount092_OFFLINE_CANDIDATE.csv`
   - Offline `0.390013`, FP `0.101781`, test selected rows `765`.
   - Reason: highest local score, but only after public confirms sequence
     family is not over-counting.

4. If public likes sequence timing but punishes count, submit:
   `submissions/seq_tcn_yolo26x_witness_3seed_thr06_nms10_cross4_snap4_rootcount082_OFFLINE_CANDIDATE.csv`
   - Offline `0.382612`, FP `0.073716`, test selected rows `712`.
   - Reason: defensive low-count sequence fallback.

5. If sequence TCN regresses, submit the alternative agreement branch:
   `submissions/yolo26x_yolo26l_agree_w4_a10_pw10_sw08_thr14_nms10_cross2_rootcount084_OFFLINE_CANDIDATE.csv`
   - Offline `0.377090`, FP `0.091709`, wins `12/13`, no root risk flags.
   - Reason: checks whether public prefers large-model agreement over the
     conservative yolo11s witness.

6. Direct yolo26x precision variants are lower priority:
   - `yolo26x_samecount_w10_am02_thr065_same8_cross4_rootrate088`: offline
     `0.372166`, total `678`.
   - `yolo26x_raw_thr065_same8_cross6_rootroundrate078`: offline `0.371622`,
     total `727`.
   - `yolo26x_samecount_w10_am02_thr065_same8_cross4_threshold`: offline
     `0.374335`, total `844`, but high count makes it risky after the
     yolo26l threshold-count public failure.

## Do Not Submit From Current Evidence

- Any dense threshold-count branch just because it wins local macro. Public
  already punished `yolo26l` threshold-count (`0.10260`).
- Audio-only, hard audio snap, global audio offset, or direct local onset
  multiplicative rescoring. On the current yolo26x anchor, nonzero audio
  rescoring regressed from `0.374335` to `0.365845`.
- Whole-video fighter swaps or current cached fighter keep/flip classifiers.
  The fighter oracle still has headroom, but the tested signals do not recover
  it.
- Simple raw-frame bbox ROI color clustering. Even oracle cluster mapping
  regressed to `0.368859`.
- More small yolo26l public threshold/root-rate tweaks before testing the
  materially different yolo26x/sequence candidates.

## Public Readout Rules

- If the first yolo26x+yolo11s agreement submit beats or roughly matches the
  `0.13849` anchor, proceed to the safer snap4 sequence TCN.
- If it drops hard, stop the queue and use `agn_038` hybrids to isolate whether
  the public hit is video-specific rather than replacing likely-private videos.
- Use at most two post-reset mask/hybrid probes unless a probe creates an
  obvious public jump.
- Always record: file, message, public score, selected rows by video, and
  whether the result changes the queue.

## Operational Guardrails

- GPU 0 stays unused. Any future GPU job must use physical GPU 1:
  `CUDA_VISIBLE_DEVICES=1` or the repo's `--cuda-visible-devices 1` wrapper.
- Verify long GPU jobs with:
  `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`.
- Every CSV must pass `tools/validate_data.py` before upload.
- Do not submit while quota says `remaining_today=0`.
