import csv
import json

import pytest

from scripts.evaluate import _validate_held_out_test
from src.evaluation import evaluate_three_systems


def test_three_way_evaluation_forwards_identical_context_and_writes_rating_templates(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({
            "id": "PC0001",
            "input_prompt": "Explain recursion",
            "target_prompt": "Explain recursion clearly",
            "category": "education",
            "intended_audience": "beginner programmer",
            "requirements": "Use a Python example",
        }) + "\n",
        encoding="utf-8",
    )
    calls = []

    def recording_baseline(prompt, **context):
        calls.append(("baseline", prompt, context))
        return "prompt-engineered output", 0.1

    def tuned(prompt, **context):
        calls.append(("tuned", prompt, context))
        return "fine-tuned output", 0.2

    output_path = tmp_path / "comparison.csv"
    rows = evaluate_three_systems(
        test_path,
        recording_baseline,
        tuned,
        output_path,
    )

    assert len(rows) == 1
    expected_context = {
        "category": "education",
        "audience": "beginner programmer",
        "output_format": "",
        "tone": "",
        "requirements": "Use a Python example",
    }
    assert calls == [
        ("baseline", "Explain recursion", expected_context),
        ("tuned", "Explain recursion", expected_context),
    ]
    assert rows[0]["original_prompt"] == rows[0]["input_prompt"]
    assert rows[0]["finetuned_output"] == "fine-tuned output"
    assert set(rows[0]) == {
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
    }

    ratings_path = tmp_path / "comparison_ratings.csv"
    mapping_path = tmp_path / "comparison_mapping.csv"
    assert ratings_path.exists()
    assert mapping_path.exists()
    with ratings_path.open(newline="", encoding="utf-8") as handle:
        rating_rows = list(csv.DictReader(handle))
    assert len(rating_rows) == 3
    for row in rating_rows:
        assert all(row[field] == "" for field in (
            "intent_preservation",
            "specificity",
            "relevance",
            "usability",
            "unsupported_assumptions",
            "reviewer_id",
            "comments",
        ))
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        mapping_rows = list(csv.DictReader(handle))
    assert len(mapping_rows) == 3
    assert {row["candidate"] for row in mapping_rows} == {"A", "B", "C"}
    assert {row["system"] for row in mapping_rows} == {
        "original_prompt",
        "prompt_engineering_baseline",
        "promptcraft_finetuned",
    }
    assert {row["id"] for row in mapping_rows} == {"PC0001"}
    systems = {row["candidate"]: row["system"] for row in mapping_rows}
    outputs = {row["candidate"]: row["candidate_output"] for row in rating_rows}
    expected_outputs = {
        candidate: {
            "original_prompt": "Explain recursion",
            "prompt_engineering_baseline": "prompt-engineered output",
            "promptcraft_finetuned": "fine-tuned output",
        }[system]
        for candidate, system in systems.items()
    }
    assert outputs == expected_outputs


def test_candidate_order_is_seeded_and_varies_by_example(tmp_path):
    test_path = tmp_path / "test.jsonl"
    records = [
        {"id": "PC0001", "input_prompt": "First", "target_prompt": "First target"},
        {"id": "PC0002", "input_prompt": "Second", "target_prompt": "Second target"},
    ]
    test_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    def generator(prompt, **context):
        return f"generated {prompt}", 0.1

    def mapping_for(output_dir, seed):
        evaluate_three_systems(
            test_path,
            generator,
            generator,
            output_dir / "comparison.csv",
            seed=seed,
        )
        with (output_dir / "comparison_mapping.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    first = mapping_for(tmp_path / "first", 17)
    second = mapping_for(tmp_path / "second", 17)
    assert first == second
    orders = [
        [row["system"] for row in first if row["id"] == record["id"]]
        for record in records
    ]
    assert orders[0] != orders[1]


def test_existing_evaluation_outputs_require_explicit_overwrite(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"id": "PC0001", "input_prompt": "First", "target_prompt": "First target"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "comparison.csv"
    output_path.write_text("existing", encoding="utf-8")

    def generator(prompt, **context):
        return "generated", 0.1

    try:
        evaluate_three_systems(test_path, generator, generator, output_path)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Existing evaluation output should require explicit overwrite")


def test_same_seed_reproduces_rating_order(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"id": "PC0001", "input_prompt": "First", "target_prompt": "First target"}) + "\n",
        encoding="utf-8",
    )

    def generator(prompt, **context):
        return "generated", 0.1

    output_path = tmp_path / "comparison.csv"
    evaluate_three_systems(test_path, generator, generator, output_path, seed=9)
    with (tmp_path / "comparison_ratings.csv").open(newline="", encoding="utf-8") as handle:
        first = [row["candidate"] for row in csv.DictReader(handle)]
    second_path = tmp_path / "comparison_second.csv"
    evaluate_three_systems(test_path, generator, generator, second_path, seed=9)
    with (tmp_path / "comparison_second_ratings.csv").open(newline="", encoding="utf-8") as handle:
        second = [row["candidate"] for row in csv.DictReader(handle)]
    assert first == second


def test_cli_requires_test_ids_to_match_split_manifest(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({"id": "PC0001", "input_prompt": "First", "target_prompt": "First target"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"ids": {"test": ["PC0002"]}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="do not match"):
        _validate_held_out_test(test_path)


def _write_valid_test_fixture(tmp_path, records=None):
    records = records or [
        {"id": "PC0001", "input_prompt": "First", "target_prompt": "First target"},
        {"id": "PC0002", "input_prompt": "Second", "target_prompt": "Second target"},
    ]
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    import hashlib
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({"ids": {"test": [record["id"] for record in records]},
                    "split_sha256": {"test": digest}}),
        encoding="utf-8",
    )
    return test_path


def test_cli_accepts_matching_test_manifest(tmp_path):
    _validate_held_out_test(_write_valid_test_fixture(tmp_path))


def test_cli_rejects_missing_manifest(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Missing split manifest"):
        _validate_held_out_test(test_path)


def test_cli_rejects_non_test_filename(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="held-out test"):
        _validate_held_out_test(train_path)


def test_cli_rejects_duplicate_test_ids(tmp_path):
    test_path = _write_valid_test_fixture(tmp_path)
    duplicate = json.loads(test_path.read_text().splitlines()[0])
    test_path.write_text(
        test_path.read_text() + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="duplicate IDs"):
        _validate_held_out_test(test_path)


def test_cli_rejects_same_ids_when_test_file_changes(tmp_path):
    test_path = _write_valid_test_fixture(tmp_path)
    records = [json.loads(line) for line in test_path.read_text().splitlines()]
    records[0]["input_prompt"] = "Modified after preparation"
    test_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="hash"):
        _validate_held_out_test(test_path)
