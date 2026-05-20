# YOLO26m After YOLO26s Hypotheses

Context: `yolo26s` full validation on the 13-video split has a reported best
grouped score of about `0.305323`. That is below the current public-transfer
anchor, `yolo11s` grouped score `0.340888`, by `-0.035565`. The main process may
be extracting `yolo26m` validation tracks on GPU 1, so this note is only for
cached validation diagnostics. No GPU work, no test generation, and no Kaggle.

## Local Evidence

- Existing notes say `yolo11s-pose` is the public anchor: local `0.340888`,
  public `0.13275`.
- `yolo11m-pose` is the warning case: local `0.363325`, public `0.12958`.
  Bigger local detector scores can overfit count/timing on the 13-video split.
- `notes/yolo26_hypotheses.md` records that this repo's extractor does not
  expose an `--end2end` switch. First YOLO26 runs therefore use Ultralytics'
  default end-to-end path.
- The ready `val_yolo26s_conf035` cache has all 13 videos and `83,201` frames.
  `val_yolo26m_conf035` currently should be treated as in-progress if any
  zero-byte or partial JSONL remains.
- Cache-level track summary at `conf=0.35`:

```text
model    complete  both_roles  red_present  blue_present  red_sw/1k  blue_sw/1k
yolo11s  13/13     0.8430      0.9326       0.9057        11.20      11.30
yolo11m  13/13     0.8693      0.9405       0.9235         8.65       7.90
yolo26s  13/13     0.7948      0.9062       0.8796        20.18      16.53
```

- Raw punch-score scale from cached tracks:

```text
model    raw_candidates  nms10x4_no_thr  p95    p99    >=0.8  >=1.15
yolo11s  127155          10372           0.694  2.748  5596   3947
yolo11m  118678          10324           0.792  3.128  5893   4239
yolo26s  109970           9849           0.639  2.706  4541   3282
```

## Why YOLO26s May Be Worse

1. End-to-end density and tracker IDs are the strongest suspect.
   `yolo26s` has fewer frames with both fighters present and many more role
   track switches. The punch heuristic uses short wrist histories per role, so
   unstable `track_id`/role continuity can erase velocity evidence even when a
   per-frame pose looks good.

2. `conf=0.35` may not mean the same thing for YOLO26.
   On the cached `conf035` extraction, `yolo26s` produces fewer raw candidates,
   fewer high-score candidates, and lower p95 heuristic score than `yolo11s`.
   A detector-conf sweep would need GPU and is outside this note, but the cached
   validation stats already say that the same extractor threshold is not
   equivalent across model families.

3. The selection score scale shifted downward.
   Reusing `yolo11s` thresholds such as `1.15` is likely too conservative for
   `yolo26s`. The grid should include lower thresholds, but a threshold-only fix
   is not enough if role coverage and track continuity remain worse.

4. Temporal grouped NMS may be suppressing noisier YOLO26 tracks differently.
   More ID switches can create adjacent candidates with changed `(fighter,
   hand)` groups. `fighter_hand` grouped NMS then keeps or suppresses a
   different event set than it does for YOLO11, even with the same windows.

5. Model size is not the first explanation.
   `yolo26m` may recover pose quality, but `yolo11m` already showed that larger
   local validation lift can fail publicly. For `yolo26m`, first require a
   better cache-quality story than "macro went up": better both-role coverage,
   lower switch rate, plausible counts, and tournament-root consistency.

## Validation-Only Checks

Run these only on completed cached validation dirs. Replace `MODEL` with
`yolo26n`, `yolo26s`, or `yolo26m`; skip a model until its directory has 13
complete JSONL files. Do not run these while `yolo26m` files are still being
written.

1. Complete-cache gate.

```bash
python3 - <<'PY'
import csv
from pathlib import Path
root = Path("data/processed/pose_tracks/val_MODEL_conf035")
videos = {r["video_key"]: int(r["frame_count"]) for r in csv.DictReader(open("data/raw/train/videos.csv"))}
keys = "agn_003,agn_004,agn_010,agn_023,agn_024,agn_025,agn_056,agn_057,agn_058,agn_069,agn_070,agn_071,agn_072".split(",")
for key in keys:
    path = root / f"{key}.jsonl"
    n = sum(1 for _ in path.open()) if path.exists() else -1
    print(key, n, videos[key], "OK" if n == videos[key] else "BAD")
PY
```

2. Narrow grouped grid around the observed YOLO26 score scale.

```bash
python3 tools/evaluate_pose_selection_variants.py \
  --tracks-dir data/processed/pose_tracks/val_MODEL_conf035 \
  --group-modes fighter_hand \
  --thresholds 0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15 \
  --same-nms-frames 8,10,12 \
  --cross-nms-frames 2,4,6 \
  --count-modes threshold,root_count,root_round_count \
  --count-multipliers 0.8,0.9,1.0 \
  --top-k 40
```

3. Track-quality audit before trusting macro.

```bash
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
root = Path("data/processed/pose_tracks/val_MODEL_conf035")
total = both = red = blue = 0
switches = defaultdict(int)
prev = defaultdict(lambda: None)
for path in sorted(root.glob("*.jsonl")):
    key = path.stem
    prev[(key, "red")] = prev[(key, "blue")] = None
    for line in path.open():
        rec = json.loads(line)
        total += 1
        fighters = rec.get("fighters") or {}
        red += fighters.get("red") is not None
        blue += fighters.get("blue") is not None
        both += fighters.get("red") is not None and fighters.get("blue") is not None
        for role in ("red", "blue"):
            f = fighters.get(role)
            tid = None if f is None else f.get("track_id")
            if tid is not None and prev[(key, role)] is not None and tid != prev[(key, role)]:
                switches[role] += 1
            if tid is not None:
                prev[(key, role)] = tid
print("both_roles", both / total, "red_present", red / total, "blue_present", blue / total)
print("switches_per_1k", {k: v * 1000 / total for k, v in switches.items()})
PY
```

4. Score-scale and density audit.

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np, sys
sys.path.insert(0, ".")
from rascar_boxing.pose_heuristic import PoseHeuristicConfig, score_pose_tracks, select_candidates
root = Path("data/processed/pose_tracks/val_MODEL_conf035")
scores = []
selected = 0
for path in sorted(root.glob("*.jsonl")):
    cands = score_pose_tracks(path, PoseHeuristicConfig(min_score=0.0))
    scores.extend(c.score for c in cands)
    selected += len(select_candidates(cands, PoseHeuristicConfig(min_score=0.0, nms_frames=10, nms_group_mode="fighter_hand", cross_nms_frames=4), None))
arr = np.array(scores, dtype=np.float32)
print("raw_candidates", len(arr), "nms10x4_no_thr", selected)
print("p50,p75,p90,p95,p99", np.percentile(arr, [50, 75, 90, 95, 99]).round(4).tolist())
for t in [0.45, 0.65, 0.85, 1.05, 1.15, 1.3]:
    print(">=", t, int((arr >= t).sum()))
PY
```

5. Temporal-context sanity only after raw grouped NMS is close.

```bash
python3 tools/evaluate_pose_temporal_context.py \
  --tracks-dir data/processed/pose_tracks/val_MODEL_conf035 \
  --features same_sum,dominance \
  --windows 4,6,8 \
  --alphas -0.2,0.0,0.1,0.2 \
  --thresholds 0.65,0.75,0.85,0.95,1.05,1.15 \
  --nms-frames 8,10,12 \
  --nms-group-modes fighter_hand \
  --cross-nms-frames 2,4 \
  --top-k 30
```

6. Cross-model agreement as a diagnostic, not a submit path.

```bash
python3 tools/evaluate_pose_model_agreement.py \
  --primary-tracks-dir data/processed/pose_tracks/val_yolo11s_conf035 \
  --secondary-tracks-dir data/processed/pose_tracks/val_MODEL_conf035 \
  --primary-name primary \
  --secondary-name secondary \
  --windows 2,4,6 \
  --agreement-alphas 0.0,0.2,0.5 \
  --primary-weights 1.0,1.2 \
  --secondary-weights 0.4,0.6,0.8 \
  --thresholds 0.6,0.8,1.0,1.2,1.4 \
  --nms-frames 8,10,12 \
  --cross-nms-frames 4 \
  --count-modes threshold,root_count \
  --count-multipliers 0.8,0.9,1.0 \
  --top-k 40
```

## Decision Rules For YOLO26m

- Do not score or compare `yolo26m` until the complete-cache gate passes for all
  13 validation videos.
- If `yolo26m` keeps `yolo26s`-like track behavior, require a very large macro
  gain before caring about threshold/context grids. Unstable tracks are a
  transfer-risk signal.
- If `yolo26m` improves both-role coverage toward `yolo11m` and lowers switch
  rate toward `yolo11s`, then compare grouped-grid macro, time score, FP
  penalty, `n_pred`, and video wins against `0.340888`.
- A `yolo26m` win that comes only from lower threshold or higher selected count
  should be treated like the failed-public `yolo11m` pattern.
- Keep `yolo26n` as a cheap compatibility/control cache if it exists; it is most
  useful for detecting whether the YOLO26 family has a systematic end-to-end
  tracker issue, not for beating `yolo11s`.
