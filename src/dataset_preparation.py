"""Prompt formatting and reproducible leakage-aware dataset preparation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .data_validation import audit_split_leakage, load_records, validate_records

SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SIMILARITY_THRESHOLD = 0.85


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
        "intended_audience": record.get("intended_audience", ""),
        "requirements": record.get("requirements", ""),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(input_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    source = Path(input_path)
    records = load_records(source)
    report = validate_records(records, str(source))
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
    leakage = audit_split_leakage(records, splits, SIMILARITY_THRESHOLD)
    (output / "leakage_report.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    realized_proportions = {
        key: len(value) / len(records) if records else 0.0
        for key, value in splits.items()
    }
    (output / "split_manifest.json").write_text(
        json.dumps({
            "dataset": {
                "filename": source.name,
                "path": str(source),
                "sha256": _sha256(source),
                "record_count": len(records),
            },
            "seed": seed,
            "split_strategy": "category-aware deterministic shuffle",
            "requested_proportions": SPLIT_PROPORTIONS,
            "realized_proportions": realized_proportions,
            "counts": {key: len(value) for key, value in splits.items()},
            "ids": {key: [row.get("id") for row in value] for key, value in splits.items()},
            "split_sha256": {
                key: _sha256(output / f"{key}.jsonl")
                for key in splits
            },
            "preprocessing": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "preserve_context_fields": ["category", "intended_audience", "requirements"],
                "output_format": "JSONL",
            },
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "report": report.to_dict(),
        "counts": {key: len(value) for key, value in splits.items()},
        "leakage": leakage,
    }
