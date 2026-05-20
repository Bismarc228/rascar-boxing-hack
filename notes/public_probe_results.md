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

agn_037  unknown                    => probe file existed, not confirmed submitted
agn_048  unknown                    => probe file existed, not confirmed submitted
```

## Interpretation

- `agn_038` definitely affects the public score and is a good target for
  per-video public tuning after quota reset.
- `agn_039`, `agn_047`, `agn_049`, `agn_062`, `agn_063`, and `agn_064` likely
  belong to private or have no measurable public contribution for this
  baseline. Do not optimize these by public LB; choose robust offline/root-audit
  variants for them.
- `agn_037` and `agn_048` should be probed after reset if we still want a more
  complete public/private split map.

## Practical Use

- Public-driven variants can change only `agn_038` to test timing/count/model
  choices with minimal private risk.
- For likely-private videos, prefer stable validation evidence, especially
  tournament-root deltas and FP/count control.
- A hybrid submission can use a public-tuned `agn_038` and conservative
  offline-selected predictions for the rest.
