"""Dataset loading and quality checks for PromptCraft AI."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {"input_prompt", "target_prompt"}
OPTIONAL_COLUMNS = {
    "id",
    "category",
    "intended_audience",
    "requirements",
    "quality_label",
    "review_status",
}


@dataclass
class ValidationReport:
    path: str
    row_count: int
    valid_rows: int
    invalid_rows: int
    missing_required_columns: list[str]
    empty_input_rows: list[int]
    empty_target_rows: list[int]
    malformed_rows: list[int]
    exact_duplicate_rows: list[int]
    near_duplicate_pairs: list[tuple[int, int]]
    categories: dict[str, int]
    review_statuses: dict[str, int]
    quality_labels: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV or JSONL records without changing the source file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL at line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} must contain an object")
                records.append(value)
        return records
    raise ValueError("Unsupported dataset format. Use .csv or .jsonl.")


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _similarity(left: str, right: str) -> float:
    left_words, right_words = set(_normalise(left).split()), set(_normalise(right).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def audit_split_leakage(
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    """Report normalized duplicate and high-lexical-similarity input pairs."""
    indexed = [
        (record, str(record.get("id", f"row-{index}")))
        for index, record in enumerate(records, 1)
        if isinstance(record, dict)
    ]
    split_by_id = {
        str(record.get("id", f"row-{index}")): split_name
        for split_name, split_records in splits.items()
        for index, record in enumerate(split_records, 1)
    }
    exact_pairs: list[dict[str, Any]] = []
    similarity_pairs: list[dict[str, Any]] = []
    for index, (left, left_id) in enumerate(indexed):
        left_prompt = _normalise(left.get("input_prompt"))
        for right, right_id in indexed[index + 1:]:
            right_prompt = _normalise(right.get("input_prompt"))
            if left_prompt and left_prompt == right_prompt:
                exact_pairs.append({
                    "left_id": left_id,
                    "right_id": right_id,
                    "similarity": 1.0,
                    "left_split": split_by_id.get(left_id),
                    "right_split": split_by_id.get(right_id),
                    "cross_split": split_by_id.get(left_id) != split_by_id.get(right_id),
                })
            score = _similarity(left_prompt, right_prompt)
            if score >= similarity_threshold:
                similarity_pairs.append({
                    "left_id": left_id,
                    "right_id": right_id,
                    "similarity": round(score, 6),
                    "left_split": split_by_id.get(left_id),
                    "right_split": split_by_id.get(right_id),
                    "cross_split": split_by_id.get(left_id) != split_by_id.get(right_id),
                })
    return {
        "similarity_threshold": similarity_threshold,
        "method": "normalized token-set Jaccard similarity; this does not detect all semantic duplicates",
        "exact_normalized_input_pairs": exact_pairs,
        "high_similarity_pairs": similarity_pairs,
        "cross_split_exact_normalized_input_pairs": [
            pair for pair in exact_pairs if pair["cross_split"]
        ],
        "cross_split_high_similarity_pairs": [
            pair for pair in similarity_pairs if pair["cross_split"]
        ],
    }


def validate_records(records: list[dict[str, Any]], path: str = "<memory>") -> ValidationReport:
    columns = set().union(*(record.keys() for record in records if isinstance(record, dict))) if records else set()
    missing = sorted(REQUIRED_COLUMNS - columns)
    empty_input = [
        i + 1 for i, r in enumerate(records)
        if not isinstance(r, dict) or not _normalise(r.get("input_prompt"))
    ]
    empty_target = [
        i + 1 for i, r in enumerate(records)
        if not isinstance(r, dict) or not _normalise(r.get("target_prompt"))
    ]
    malformed = [
        i + 1 for i, r in enumerate(records)
        if not isinstance(r, dict) or any(key not in r for key in REQUIRED_COLUMNS)
    ]
    seen: dict[tuple[str, str], int] = {}
    exact_duplicates: list[int] = []
    for i, record in enumerate(records, 1):
        safe_record = record if isinstance(record, dict) else {}
        key = (_normalise(safe_record.get("input_prompt")), _normalise(safe_record.get("target_prompt")))
        if key in seen:
            exact_duplicates.append(i)
        else:
            seen[key] = i
    near_pairs: list[tuple[int, int]] = []
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            left_prompt = left.get("input_prompt", "") if isinstance(left, dict) else ""
            right_prompt = records[j].get("input_prompt", "") if isinstance(records[j], dict) else ""
            if _similarity(left_prompt, right_prompt) >= 0.85:
                near_pairs.append((i + 1, j + 1))
    categories = Counter(_normalise(r.get("category")) or "uncategorized" for r in records if isinstance(r, dict))
    statuses = Counter(_normalise(r.get("review_status")) or "missing" for r in records if isinstance(r, dict))
    labels = Counter(_normalise(r.get("quality_label")) or "missing" for r in records if isinstance(r, dict))
    invalid_count = len(set(empty_input + empty_target + malformed))
    warnings: list[str] = []
    if statuses and any(status not in {"reviewed", "approved"} for status in statuses):
        warnings.append("Dataset contains records that are not marked reviewed/approved; labels are not ground truth.")
    if near_pairs:
        warnings.append(f"Found {len(near_pairs)} near-duplicate input pairs; keep related examples in one split.")
    if len(records) < 100:
        warnings.append("The dataset is small for reliable fine-tuning; treat results as a prototype.")
    return ValidationReport(
        path=path,
        row_count=len(records),
        valid_rows=len(records) - invalid_count,
        invalid_rows=invalid_count,
        missing_required_columns=missing,
        empty_input_rows=empty_input,
        empty_target_rows=empty_target,
        malformed_rows=malformed,
        exact_duplicate_rows=exact_duplicates,
        near_duplicate_pairs=near_pairs,
        categories=dict(categories),
        review_statuses=dict(statuses),
        quality_labels=dict(labels),
        warnings=warnings,
    )


def validate_dataset(path: str | Path) -> ValidationReport:
    records = load_records(path)
    report = validate_records(records, str(path))
    if report.missing_required_columns:
        raise ValueError(f"Missing required columns: {', '.join(report.missing_required_columns)}")
    return report
