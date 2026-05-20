# RASCAR Boxing Baseline

The current baseline is intentionally small and reproducible. It validates the
downloaded data, creates a valid submission, implements the local metric, and
contains a script version of the YOLO-pose role-assignment notebook.

## Commands

Validate dataset and the generated submission:

```bash
python3 tools/validate_data.py --data-root data/raw --submission submission.csv --check-video-props
```

Generate the first submission:

```bash
python3 tools/make_baseline_submission.py \
  --data-root data/raw \
  --output submission.csv \
  --strategy sample
```

Create a deterministic fight-level validation split:

```bash
python3 tools/make_validation_split.py \
  --videos-csv data/raw/train/videos.csv \
  --output-dir data/processed/splits/fight_seed42
```

Sanity-check the metric by scoring train labels against themselves:

```bash
python3 tools/score_predictions.py \
  --ground-truth data/raw/train/punches.csv \
  --predictions data/raw/train/punches.csv \
  --output-json data/processed/train_self_score.json
```

Run the ported YOLO-pose baseline on one video after installing the optional
YOLO dependencies:

```bash
python3 tools/run_pose_baseline.py \
  --data-root data/raw \
  --video-key agn_037 \
  --model models/yolo11n-pose.pt \
  --tracks-out data/processed/pose_tracks/agn_037.jsonl \
  --output data/processed/pose_tracks/agn_037_annotated.mp4 \
  --max-frames 300
```

The local environment has been checked with `torch`, `torchvision`,
`ultralytics`, and `lap`. A one-frame smoke run wrote
`data/processed/pose_tracks/smoke_agn_037.jsonl`.
