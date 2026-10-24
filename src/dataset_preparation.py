"""Prompt formatting and reproducible leakage-aware dataset preparation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .data_validation import load_records, validate_records


def format_example(record: dict[str, Any]) -> dict[str, Any]:
    context = [
        f"Category: {record.get('category', 'general')}",
        f"Target audience: {record.get('intended_audience', 'not specified')}",
        f"Additional requirements: {record.get('requirements', 'none')}",
    ]
    user = f"Optimize the following user request while preserving its intent.\n\n{record['input_prompt']}\n\n" + "\n".join(context)
    return {
        "id": record.get("id"),
        "input_prompt": record["input_prompt"],
        "target_prompt": record["target_prompt"],
        "text": f"<|user|>\n{user}\n<|assistant|>\n{record['target_prompt']}",
        "category": record.get("category", "general"),
    }


def split_records(records: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Split by category using a deterministic shuffle; duplicate checks run before this."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get("category", "general")), []).append(record)
    splits = {"train": [], "validation": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        test_count = max(1, round(n * 0.15))
        validation_count = max(1, round(n * 0.15)) if n >= 4 else 0
        splits["test"].extend(group[:test_count])
        splits["validation"].extend(group[test_count:test_count + validation_count])
        splits["train"].extend(group[test_count + validation_count:])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def prepare_dataset(input_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    records = load_records(input_path)
    report = validate_records(records, str(input_path))
    if report.missing_required_columns or report.invalid_rows:
        raise ValueError(f"Dataset is invalid: {report.to_dict()}")
    if report.exact_duplicate_rows:
        raise ValueError("Exact duplicate examples must be removed before splitting.")
    splits = split_records(records, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(format_example(row), ensure_ascii=False) + "\n")
    (output / "quality_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (output / "split_manifest.json").write_text(
        json.dumps({"seed": seed, "counts": {key: len(value) for key, value in splits.items()}}, indent=2),
        encoding="utf-8",
    )
    return {"report": report.to_dict(), "counts": {key: len(value) for key, value in splits.items()}}
