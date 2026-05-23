"""Shared constants for dataset validation and scoring."""

FPS = 30.0

SUBMISSION_COLUMNS = [
    "id",
    "video_id",
    "agn_index",
    "video_key",
    "frame",
    "fighter",
    "punch_type",
    "hand",
    "target",
    "effectiveness",
    "clear",
]

ALLOWED_VALUES = {
    "fighter": {"red", "blue"},
    "punch_type": {"jab", "cross", "hook", "uppercut"},
    "hand": {"left", "right"},
    "target": {"head", "body"},
    "effectiveness": {"landed", "blocked", "miss"},
    "clear": {"true", "false"},
}

EFFECTIVENESS_METRIC_MAP = {
    "landed": "hit",
    "blocked": "block",
    "miss": "miss",
}

