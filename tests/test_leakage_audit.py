from src.data_validation import audit_split_leakage


def test_duplicate_pairs_and_cross_split_leakage_are_reported():
    records = [
        {"id": "a", "input_prompt": "Explain recursion with Python examples for beginners"},
        {"id": "b", "input_prompt": " explain recursion with Python examples for beginners "},
        {"id": "c", "input_prompt": "Explain recursion with Python examples for beginners today"},
    ]
    splits = {
        "train": [records[0]],
        "validation": [records[1]],
        "test": [records[2]],
    }

    report = audit_split_leakage(records, splits)

    assert report["exact_normalized_input_pairs"] == [{
        "left_id": "a",
        "right_id": "b",
        "similarity": 1.0,
        "left_split": "train",
        "right_split": "validation",
        "cross_split": True,
    }]
    assert any(
        pair["left_id"] == "a"
        and pair["right_id"] == "c"
        and pair["cross_split"]
        for pair in report["high_similarity_pairs"]
    )
    assert report["cross_split_exact_normalized_input_pairs"]
    assert report["cross_split_high_similarity_pairs"]
