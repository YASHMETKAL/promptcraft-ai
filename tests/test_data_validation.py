import pytest

from src.data_validation import load_records, validate_records


def test_supplied_dataset_is_nonempty_and_valid():
    records = load_records("data/promptcraft_custom_dataset.csv")
    report = validate_records(records)
    assert report.row_count == 82
    assert report.invalid_rows == 0
    assert report.missing_required_columns == []


def test_empty_prompt_is_reported():
    report = validate_records([{"input_prompt": "", "target_prompt": "answer"}])
    assert report.empty_input_rows == [1]
    assert report.invalid_rows == 1


def test_missing_file_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_records("does-not-exist.csv")
