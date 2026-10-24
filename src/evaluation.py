"""Evaluation utilities. Subjective quality scores are supplied by human raters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable


def evaluate_models(
    test_path: str | Path,
    baseline_generator: Callable[[str], tuple[str, float]],
    finetuned_generator: Callable[[str], tuple[str, float]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(test_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            base, base_time = baseline_generator(record["input_prompt"])
            tuned, tuned_time = finetuned_generator(record["input_prompt"])
            rows.append({
                "id": record.get("id"),
                "input_prompt": record["input_prompt"],
                "reference_prompt": record["target_prompt"],
                "baseline_output": base,
                "finetuned_output": tuned,
                "baseline_inference_seconds": base_time,
                "finetuned_inference_seconds": tuned_time,
                "baseline_clarity": "",
                "finetuned_clarity": "",
                "baseline_specificity": "",
                "finetuned_specificity": "",
                "baseline_intent_preservation": "",
                "finetuned_intent_preservation": "",
                "rating_note": "Fill 1-5 human rubric scores; blank means not rated.",
            })
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
