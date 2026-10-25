"""Evaluation utilities. Subjective quality scores are supplied by human raters."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable


Generator = Callable[..., tuple[str, float]]
THREE_WAY_COLUMNS = [
    "id",
    "input_prompt",
    "category",
    "intended_audience",
    "requirements",
    "original_prompt",
    "prompt_engineering_output",
    "finetuned_output",
    "prompt_engineering_inference_seconds",
    "finetuned_inference_seconds",
]
RATING_COLUMNS = [
    "id",
    "input_prompt",
    "category",
    "intended_audience",
    "requirements",
    "candidate",
    "candidate_output",
    "intent_preservation",
    "specificity",
    "relevance",
    "usability",
    "unsupported_assumptions",
    "reviewer_id",
    "comments",
]


def generate_prompt_engineering_baseline(
    original_prompt: str,
    category: str = "General",
    audience: str = "",
    output_format: str = "",
    tone: str = "",
    requirements: str = "",
) -> tuple[str, float]:
    """Create a deterministic non-fine-tuned prompt-engineering baseline."""
    if not original_prompt.strip():
        raise ValueError("Original prompt cannot be empty.")
    start = time.perf_counter()
    result = (
        "Rewrite the following request as a clear, standalone prompt for an AI assistant. "
        "Preserve the user's intent and include only relevant context and constraints. "
        f"Request: {original_prompt.strip()}\n"
        f"Category: {category}\n"
        f"Target audience: {audience or 'not specified'}\n"
        f"Output format: {output_format or 'not specified'}\n"
        f"Tone: {tone or 'not specified'}\n"
        f"Additional requirements: {requirements or 'none'}"
    )
    return result, time.perf_counter() - start


def _record_context(record: dict[str, Any]) -> dict[str, str]:
    return {
        "category": record.get("category", "General"),
        "audience": record.get("intended_audience", ""),
        "output_format": record.get("output_format", ""),
        "tone": record.get("tone", ""),
        "requirements": record.get("requirements", ""),
    }


def evaluate_three_systems(
    test_path: str | Path,
    prompt_engineering_generator: Generator,
    finetuned_generator: Generator,
    output_path: str | Path,
    ratings_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Generate original, prompt-engineering, and fine-tuned comparison artifacts."""
    destination = Path(output_path)
    if ratings_path is None:
        ratings_path = destination.with_name(f"{destination.stem}_ratings.csv")
    if mapping_path is None:
        mapping_path = destination.with_name(f"{destination.stem}_mapping.csv")
    destinations = [destination, Path(ratings_path), Path(mapping_path)]
    if not overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                f"Evaluation output already exists: {', '.join(existing)}. "
                "Choose a new --output path or pass --overwrite."
            )
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    rating_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, str]] = []
    with Path(test_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            context = _record_context(record)
            prompt = record["input_prompt"]
            baseline, baseline_time = prompt_engineering_generator(prompt, **context)
            tuned, tuned_time = finetuned_generator(prompt, **context)
            row = {
                "id": record.get("id"),
                "input_prompt": prompt,
                "category": context["category"],
                "intended_audience": context["audience"],
                "requirements": context["requirements"],
                "original_prompt": prompt,
                "prompt_engineering_output": baseline,
                "finetuned_output": tuned,
                "prompt_engineering_inference_seconds": baseline_time,
                "finetuned_inference_seconds": tuned_time,
            }
            rows.append(row)
            candidates = [
                ("original_prompt", prompt),
                ("prompt_engineering_baseline", baseline),
                ("promptcraft_finetuned", tuned),
            ]
            rng.shuffle(candidates)
            candidate_labels = ("A", "B", "C")
            for candidate, (system, candidate_output) in zip(candidate_labels, candidates):
                mapping_rows.append({
                    "id": str(record.get("id")),
                    "candidate": candidate,
                    "system": system,
                })
                rating_rows.append({
                    "id": record.get("id"),
                    "input_prompt": prompt,
                    "category": context["category"],
                    "intended_audience": context["audience"],
                    "requirements": context["requirements"],
                    "candidate": candidate,
                    "candidate_output": candidate_output,
                    "intent_preservation": "",
                    "specificity": "",
                    "relevance": "",
                    "usability": "",
                    "unsupported_assumptions": "",
                    "reviewer_id": "",
                    "comments": "",
                })
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=THREE_WAY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if ratings_path is None:
        ratings_path = destination.with_name(f"{destination.stem}_ratings.csv")
    if mapping_path is None:
        mapping_path = destination.with_name(f"{destination.stem}_mapping.csv")
    for path, fieldnames, values in (
        (Path(ratings_path), RATING_COLUMNS, rating_rows),
        (Path(mapping_path), ["id", "candidate", "system"], mapping_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(values)
    return rows


def evaluate_models(
    test_path: str | Path,
    baseline_generator: Generator,
    finetuned_generator: Generator,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(test_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            context = {
                "category": record.get("category", "General"),
                "audience": record.get("intended_audience", ""),
                "output_format": record.get("output_format", ""),
                "tone": record.get("tone", ""),
                "requirements": record.get("requirements", ""),
            }
            base, base_time = baseline_generator(record["input_prompt"], **context)
            tuned, tuned_time = finetuned_generator(record["input_prompt"], **context)
            rows.append({
                "id": record.get("id"),
                "input_prompt": record["input_prompt"],
                "reference_prompt": record["target_prompt"],
                "category": context["category"],
                "intended_audience": context["audience"],
                "requirements": context["requirements"],
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
