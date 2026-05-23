#!/usr/bin/env python3
"""Evaluate simple fight-level source-switch stumps with group OOF."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_FEATURES = [
    "n_pred",
    "pred_per_min",
    "base_count",
    "count_delta_vs_base",
    "base_le15",
    "source_le15",
    "base_med_gap",
    "source_med_gap",
]


@dataclass(frozen=True)
class Rule:
    source: str
    feature: str
    op: str
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.source}:{self.feature}{self.op}{self.threshold:g}"


@dataclass(frozen=True)
class RuleResult:
    rule: Rule | None
    score: float
    base_score: float
    delta: float
    switches: int
    wins: int
    losses: int
    choices: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-table", type=Path, required=True)
    parser.add_argument("--test-table", type=Path)
    parser.add_argument("--base-source", default="seq_motion")
    parser.add_argument("--sources", default="", help="Comma-separated override sources")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-train-regression",
        action="store_true",
        help="Allow fold rule selection below the no-op base score.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.validation_table)
    rows = [row for row in rows if row.get("score")]
    features = [item for item in args.features.split(",") if item]
    sources = [item for item in args.sources.split(",") if item]
    if not sources:
        sources = sorted({row["source"] for row in rows if row["source"] != args.base_source})

    by_video = group_by_video(rows)
    valid_keys = [
        key
        for key, video_rows in sorted(by_video.items())
        if args.base_source in {row["source"] for row in video_rows}
    ]
    rows = [row for row in rows if row["video_key"] in set(valid_keys)]
    by_video = group_by_video(rows)
    groups = {key: fight_group(by_video[key][0]) for key in valid_keys}

    print(f"base={args.base_source} base_score={base_mean(by_video, args.base_source):.6f}")
    print("in_sample_top")
    all_results = evaluate_rules(by_video, args.base_source, sources, features)
    for item in all_results[: args.top_k]:
        print(format_result(item, args.base_source))

    oof_result, fold_lines = evaluate_group_oof(
        by_video,
        groups,
        args.base_source,
        sources,
        features,
        allow_train_regression=args.allow_train_regression,
    )
    print("group_oof")
    for line in fold_lines:
        print(line)
    print(format_result(oof_result, args.base_source, label="OOF_SELECTED"))

    if args.test_table:
        test_rows = read_rows(args.test_table)
        test_by_video = group_by_video(test_rows)
        best_full = all_results[0].rule
        print("test_full_data_rule")
        if best_full is None:
            print("rule=BASE choices=" + ",".join(f"{key}:{args.base_source}" for key in sorted(test_by_video)))
        else:
            choices = apply_rule_choices(test_by_video, args.base_source, best_full)
            selected = [key for key, source in choices.items() if source != args.base_source]
            print(f"rule={best_full.label} selected={','.join(selected)}")
            print("choices=" + ",".join(f"{key}:{source}" for key, source in choices.items()))
    return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def group_by_video(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["video_key"], []).append(row)
    return dict(sorted(grouped.items()))


def fight_group(row: dict[str, str]) -> str:
    return f"{row['data_root']}|{row['fight_index']}"


def row_by_source(video_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["source"]: row for row in video_rows}


def base_mean(by_video: dict[str, list[dict[str, str]]], base_source: str) -> float:
    return float(
        np.mean([float(row_by_source(rows)[base_source]["score"]) for rows in by_video.values()])
    )


def evaluate_rules(
    by_video: dict[str, list[dict[str, str]]],
    base_source: str,
    sources: list[str],
    features: list[str],
) -> list[RuleResult]:
    rules: list[Rule | None] = [None]
    for source in sources:
        source_rows = [
            row_by_source(video_rows)[source]
            for video_rows in by_video.values()
            if source in row_by_source(video_rows)
        ]
        for feature in features:
            values = sorted({float(row[feature]) for row in source_rows if row.get(feature) not in {"", None}})
            for threshold in values:
                rules.append(Rule(source=source, feature=feature, op=">=", threshold=threshold))
                rules.append(Rule(source=source, feature=feature, op="<=", threshold=threshold))
    results = [evaluate_rule(by_video, base_source, rule) for rule in rules]
    return sorted(results, key=lambda item: (item.score, item.delta, -item.switches), reverse=True)


def evaluate_rule(
    by_video: dict[str, list[dict[str, str]]],
    base_source: str,
    rule: Rule | None,
) -> RuleResult:
    scores = []
    base_scores = []
    choices: dict[str, str] = {}
    switches = wins = losses = 0
    for key, video_rows in by_video.items():
        by_source = row_by_source(video_rows)
        base_row = by_source[base_source]
        source = base_source
        if rule is not None and rule.source in by_source and rule_passes(by_source[rule.source], rule):
            source = rule.source
        choice_row = by_source[source]
        base_score = float(base_row["score"])
        score = float(choice_row["score"])
        scores.append(score)
        base_scores.append(base_score)
        choices[key] = source
        if source != base_source:
            switches += 1
            wins += int(score > base_score)
            losses += int(score < base_score)
    score = float(np.mean(scores))
    base_score = float(np.mean(base_scores))
    return RuleResult(
        rule=rule,
        score=score,
        base_score=base_score,
        delta=score - base_score,
        switches=switches,
        wins=wins,
        losses=losses,
        choices=choices,
    )


def rule_passes(row: dict[str, str], rule: Rule) -> bool:
    value = float(row[rule.feature])
    if rule.op == ">=":
        return value >= rule.threshold
    if rule.op == "<=":
        return value <= rule.threshold
    raise ValueError(rule.op)


def evaluate_group_oof(
    by_video: dict[str, list[dict[str, str]]],
    groups: dict[str, str],
    base_source: str,
    sources: list[str],
    features: list[str],
    *,
    allow_train_regression: bool,
) -> tuple[RuleResult, list[str]]:
    scores = []
    base_scores = []
    choices: dict[str, str] = {}
    fold_lines = []
    for group in sorted(set(groups.values())):
        valid_keys = {key for key, value in groups.items() if value == group}
        train = {key: rows for key, rows in by_video.items() if key not in valid_keys}
        valid = {key: rows for key, rows in by_video.items() if key in valid_keys}
        train_results = evaluate_rules(train, base_source, sources, features)
        if allow_train_regression:
            selected = train_results[0]
        else:
            selected = max(
                train_results,
                key=lambda item: (item.score >= item.base_score, item.score, -item.switches),
            )
            if selected.score < selected.base_score:
                selected = evaluate_rule(train, base_source, None)
        applied = evaluate_rule(valid, base_source, selected.rule)
        fold_lines.append(
            "fold="
            + group
            + f" train_rule={rule_label(selected.rule)}"
            + f" train_score={selected.score:.6f}"
            + f" valid_score={applied.score:.6f}"
            + f" valid_base={applied.base_score:.6f}"
            + " choices="
            + ",".join(f"{key}:{source}" for key, source in applied.choices.items())
        )
        for key, video_rows in valid.items():
            by_source = row_by_source(video_rows)
            source = applied.choices[key]
            scores.append(float(by_source[source]["score"]))
            base_scores.append(float(by_source[base_source]["score"]))
            choices[key] = source
    score = float(np.mean(scores))
    base_score = float(np.mean(base_scores))
    switches = sum(source != base_source for source in choices.values())
    wins = losses = 0
    for key, source in choices.items():
        if source == base_source:
            continue
        by_source = row_by_source(by_video[key])
        delta = float(by_source[source]["score"]) - float(by_source[base_source]["score"])
        wins += int(delta > 0)
        losses += int(delta < 0)
    return (
        RuleResult(
            rule=None,
            score=score,
            base_score=base_score,
            delta=score - base_score,
            switches=switches,
            wins=wins,
            losses=losses,
            choices=choices,
        ),
        fold_lines,
    )


def apply_rule_choices(
    by_video: dict[str, list[dict[str, str]]],
    base_source: str,
    rule: Rule,
) -> dict[str, str]:
    choices = {}
    for key, video_rows in sorted(by_video.items()):
        by_source = row_by_source(video_rows)
        source = base_source
        if rule.source in by_source and rule_passes(by_source[rule.source], rule):
            source = rule.source
        choices[key] = source
    return choices


def rule_label(rule: Rule | None) -> str:
    return "BASE" if rule is None else rule.label


def format_result(result: RuleResult, base_source: str, label: str | None = None) -> str:
    rule = label or rule_label(result.rule)
    return (
        f"rule={rule} score={result.score:.6f} "
        f"base={result.base_score:.6f} delta={result.delta:.6f} "
        f"switches={result.switches} wins={result.wins} losses={result.losses} "
        "choices="
        + ",".join(f"{key}:{source}" for key, source in result.choices.items() if source != base_source)
    )


if __name__ == "__main__":
    raise SystemExit(main())
