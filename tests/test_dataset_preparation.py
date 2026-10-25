import json

from src.dataset_preparation import format_example, prepare_dataset, split_records


def test_format_preserves_target():
    record = {
        "input_prompt": "Explain AI",
        "target_prompt": "Explain AI to a beginner",
        "category": "education",
        "intended_audience": "student",
        "requirements": "Use an example",
    }
    formatted = format_example(record)
    assert "Explain AI" in formatted["text"]
    assert formatted["target_prompt"] in formatted["text"]
    assert formatted["intended_audience"] == "student"
    assert formatted["requirements"] == "Use an example"


def test_split_is_reproducible():
    records = [{"id": str(i), "input_prompt": f"p{i}", "target_prompt": f"t{i}", "category": "x"} for i in range(20)]
    first = split_records(records, 7)
    second = split_records(records, 7)
    assert [[row["id"] for row in first[key]] for key in first] == [[row["id"] for row in second[key]] for key in second]


def test_manifest_is_deterministic_and_accounts_for_all_ids(tmp_path):
    source = tmp_path / "dataset.csv"
    source.write_text(
        "id,category,input_prompt,target_prompt\n"
        "a,x,Explain A,Explain A clearly\n"
        "b,x,Explain B,Explain B clearly\n"
        "c,x,Explain C,Explain C clearly\n"
        "d,x,Explain D,Explain D clearly\n",
        encoding="utf-8",
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    prepare_dataset(source, first_dir, seed=7)
    prepare_dataset(source, second_dir, seed=7)
    first = json.loads((first_dir / "split_manifest.json").read_text())
    second = json.loads((second_dir / "split_manifest.json").read_text())
    assert first == second
    assigned = [item for values in first["ids"].values() for item in values]
    assert set(assigned) == {"a", "b", "c", "d"}
    assert len(assigned) == len(set(assigned))
    assert first["dataset"]["sha256"]
    assert first["requested_proportions"] == {"train": 0.7, "validation": 0.15, "test": 0.15}


def test_manifest_hash_changes_when_source_changes(tmp_path):
    source = tmp_path / "dataset.csv"
    source.write_text(
        "id,category,input_prompt,target_prompt\n"
        "a,x,Explain A,Explain A clearly\n",
        encoding="utf-8",
    )
    first_dir = tmp_path / "first"
    prepare_dataset(source, first_dir)
    first = json.loads((first_dir / "split_manifest.json").read_text())
    source.write_text(source.read_text() + "b,x,Explain B,Explain B clearly\n", encoding="utf-8")
    second_dir = tmp_path / "second"
    prepare_dataset(source, second_dir)
    second = json.loads((second_dir / "split_manifest.json").read_text())
    assert first["dataset"]["sha256"] != second["dataset"]["sha256"]
