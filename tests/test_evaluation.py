import json
import csv

from src.evaluation import evaluate_models


def test_evaluation_forwards_context_to_both_models(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({
            "id": "PC0001",
            "input_prompt": "Explain recursion",
            "target_prompt": "Explain recursion to a beginner",
            "category": "education",
            "intended_audience": "beginner programmer",
            "requirements": "Use a Python example",
        }) + "\n",
        encoding="utf-8",
    )
    calls = []

    def generator(name):
        def generate(prompt, **context):
            calls.append((name, prompt, context))
            return f"{name} output", 0.1
        return generate

    output_path = tmp_path / "evaluation.csv"
    rows = evaluate_models(test_path, generator("base"), generator("tuned"), output_path)

    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]
    assert calls[0][2] == {
        "category": "education",
        "audience": "beginner programmer",
        "output_format": "",
        "tone": "",
        "requirements": "Use a Python example",
    }
    assert rows[0]["category"] == "education"
    assert rows[0]["intended_audience"] == "beginner programmer"
    assert rows[0]["requirements"] == "Use a Python example"
    with output_path.open(newline="", encoding="utf-8") as handle:
        columns = csv.DictReader(handle).fieldnames
    assert columns is not None
    assert {
        "category",
        "intended_audience",
        "requirements",
        "baseline_clarity",
        "finetuned_clarity",
    }.issubset(columns)


def test_evaluation_preserves_rating_columns_as_blank(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(
        json.dumps({
            "id": "PC0002",
            "input_prompt": "Explain databases",
            "target_prompt": "Explain databases clearly",
            "category": "education",
        }) + "\n",
        encoding="utf-8",
    )

    def generator(prompt, **context):
        return "generated", 0.2

    output_path = tmp_path / "evaluation.csv"
    rows = evaluate_models(test_path, generator, generator, output_path)

    assert rows[0]["baseline_clarity"] == ""
    assert rows[0]["finetuned_clarity"] == ""
    assert rows[0]["baseline_specificity"] == ""
    assert rows[0]["finetuned_specificity"] == ""
    assert rows[0]["baseline_intent_preservation"] == ""
    assert rows[0]["finetuned_intent_preservation"] == ""
