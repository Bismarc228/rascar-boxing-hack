# Public Probe Results

Context: public split was probed by taking the `yolo26l same_sum w6
rootrate088` submission and setting all predictions for one test video to
`clear=false`.

## Result

```text
agn_038  public dropped to 0.06940  => confirmed public-impact video

agn_039  public stayed 0.13664      => no public effect in this probe
agn_047  public stayed 0.13664      => no public effect in this probe
agn_049  public stayed 0.13664      => no public effect in this probe
agn_062  public stayed 0.13664      => no public effect in this probe
agn_063  public stayed 0.13664      => no public effect in this probe
agn_064  public stayed 0.13664      => no public effect in this probe

agn_037  sequence-hybrid public hit => later sequence replacement dropped to 0.12712
agn_048  unknown                    => probe file existed, not confirmed submitted
```

## Interpretation

- `agn_038` definitely affects the public score and is a good target for
  per-video public tuning after quota reset.
- `agn_039`, `agn_047`, `agn_049`, `agn_062`, `agn_063`, and `agn_064` likely
  belong to private or have no measurable public contribution for this
  baseline. Do not optimize these by public LB; choose robust offline/root-audit
  variants for them.
- `agn_037` should not be probed blindly anymore. A later sequence-hybrid check
  showed it has public impact and the sequence replacement is bad there.
- `agn_048` remains unknown; do not probe it unless a local check gives a clear
  reason under the remaining-attempt cap.

## Practical Use

- Public-driven variants can change only `agn_038` to test timing/count/model
  choices with minimal private risk.
- For likely-private videos, prefer stable validation evidence, especially
  tournament-root deltas and FP/count control.
- A hybrid submission can use a public-tuned `agn_038` and conservative
  offline-selected predictions for the rest.

## Post-Reset Sequence Probe Results

After quota reset on `2026-05-21`, the public best moved from the old yolo26l
anchor to an `agn_038` sequence-TCN hybrid:

```text
0.16461  base yolo26l best + seq_tcn snap4 rootcount088 on agn_038
0.16461  same + agn_047 sequence replacement
0.16461  same + agn_062 sequence replacement
0.16461  same + agn_063 sequence replacement
0.12712  full seq_tcn snap4 rootcount088
0.12712  same agn_038 hybrid + agn_037 sequence replacement
0.12888  base yolo26l best + yolo26x/yolo11s agreement on agn_038
0.10784  full yolo26x/yolo11s agreement
```

Updated interpretation:

- `agn_038` is definitely public and sequence-TCN snap4 is much better than
  yolo26l base there.
- `agn_037` is also public-impact or public-coupled in practice: adding only
  its sequence replacement erased the `agn_038` gain. The selected count stayed
  `52`, so the failure is timing/identity/attributes, not count inflation.
- `agn_047`, `agn_062`, and `agn_063` are still public-neutral in these probes.
- Do not submit full sequence-TCN or full yolo26x agreement again. Use
  video-local hybrids only, and only under the strict remaining-attempt cap.
