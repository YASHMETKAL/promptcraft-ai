from src.dataset_preparation import format_example, split_records


def test_format_preserves_target():
    record = {"input_prompt": "Explain AI", "target_prompt": "Explain AI to a beginner", "category": "education"}
    formatted = format_example(record)
    assert "Explain AI" in formatted["text"]
    assert formatted["target_prompt"] in formatted["text"]


def test_split_is_reproducible():
    records = [{"id": str(i), "input_prompt": f"p{i}", "target_prompt": f"t{i}", "category": "x"} for i in range(20)]
    first = split_records(records, 7)
    second = split_records(records, 7)
    assert [[row["id"] for row in first[key]] for key in first] == [[row["id"] for row in second[key]] for key in second]
