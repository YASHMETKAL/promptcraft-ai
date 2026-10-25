import json
import hashlib
from types import SimpleNamespace

import pytest

from scripts.train import (
    build_training_kwargs,
    build_training_data_files,
    validate_output_and_resume,
    validate_training_data,
    write_experiment_metadata,
)


def _prepared_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    records = {
        "train": [{"id": "a", "input_prompt": "A", "target_prompt": "A target"}],
        "validation": [{"id": "b", "input_prompt": "B", "target_prompt": "B target"}],
        "test": [{"id": "c", "input_prompt": "C", "target_prompt": "C target"}],
    }
    hashes = {}
    for split, rows in records.items():
        path = data_dir / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data_dir / "split_manifest.json").write_text(json.dumps({
        "dataset": {"filename": "unavailable.csv", "path": "unavailable.csv", "sha256": "unknown"},
        "split_sha256": hashes,
        "ids": {split: [row["id"] for row in rows] for split, rows in records.items()},
        "counts": {split: len(rows) for split, rows in records.items()},
    }), encoding="utf-8")
    return data_dir


def test_training_kwargs_use_matching_epoch_evaluation_and_save():
    config = {
        "num_train_epochs": 3,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "seed": 42,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    fields = {
        "eval_strategy",
        "save_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "bf16",
        "fp16",
    }
    kwargs = build_training_kwargs(config, fields, False, True, "output")
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["save_strategy"] == "epoch"
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False


def test_training_kwargs_reject_mismatched_save_and_eval_intervals():
    config = {
        "num_train_epochs": 3,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "seed": 42,
        "eval_strategy": "epoch",
        "save_strategy": "steps",
    }
    with pytest.raises(ValueError, match="save_strategy"):
        build_training_kwargs(config, {"eval_strategy", "save_strategy"}, False, False, "output")


def test_metadata_records_validation_only_checkpoint_selection(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "split_manifest.json").write_text(json.dumps({
        "dataset": {"filename": "dataset.csv", "sha256": "source-hash"},
        "split_sha256": {"train": "train-hash", "validation": "validation-hash", "test": "test-hash"},
        "ids": {"train": ["a"], "validation": ["b"], "test": ["c"]},
    }), encoding="utf-8")
    trainer = SimpleNamespace(state=SimpleNamespace(
        best_model_checkpoint="output/checkpoint-2",
        best_metric=1.23,
    ))
    config = {
        "model_name": "test-model",
        "seed": 42,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_seq_length": 512,
        "num_train_epochs": 3,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
    }
    metadata = write_experiment_metadata(tmp_path / "output", config, data_dir, trainer)
    assert metadata["dataset"]["split_sha256"]["test"] == "test-hash"
    assert metadata["checkpoint_selection"] == {
        "evaluation_split": "validation",
        "metric": "eval_loss",
        "direction": "minimize",
        "selected_checkpoint": "output/checkpoint-2",
        "best_validation_loss": 1.23,
    }
    assert metadata["dataset"]["train_record_count"] == 1
    assert metadata["dataset"]["validation_record_count"] == 1


def test_valid_preflight_checks_all_split_hashes(tmp_path):
    data_dir = _prepared_fixture(tmp_path)
    manifest = validate_training_data(data_dir)
    assert manifest["split_sha256"]["test"]
    assert manifest["_source_verification"] == "unavailable"


def test_training_data_files_exclude_test_split(tmp_path):
    files = build_training_data_files(tmp_path)
    assert set(files) == {"train", "validation"}
    assert "test" not in files


def test_preflight_rejects_modified_split_hash(tmp_path):
    data_dir = _prepared_fixture(tmp_path)
    (data_dir / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_training_data(data_dir)


def test_preflight_rejects_duplicate_ids(tmp_path):
    data_dir = _prepared_fixture(tmp_path)
    path = data_dir / "train.jsonl"
    path.write_text(path.read_text() + path.read_text(), encoding="utf-8")
    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    manifest["split_sha256"]["train"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["ids"]["train"] = ["a", "a"]
    manifest["counts"]["train"] = 2
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_training_data(data_dir)


def test_preflight_rejects_overlapping_ids(tmp_path):
    data_dir = _prepared_fixture(tmp_path)
    path = data_dir / "validation.jsonl"
    path.write_text(json.dumps({"id": "a", "input_prompt": "B", "target_prompt": "B target"}) + "\n")
    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    manifest["split_sha256"]["validation"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["ids"]["validation"] = ["a"]
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        validate_training_data(data_dir)


def test_output_defaults_to_non_resume_and_rejects_nonempty_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    validate_output_and_resume(tmp_path / "new-output", None, False)
    with pytest.raises(FileExistsError, match="not empty"):
        validate_output_and_resume(output, None, False)


def test_invalid_resume_path_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="trainer_state.json"):
        validate_output_and_resume(tmp_path / "output", str(tmp_path / "missing"), False)


def test_metadata_records_actual_trainer_settings(tmp_path):
    data_dir = _prepared_fixture(tmp_path)
    trainer = SimpleNamespace(state=SimpleNamespace(best_model_checkpoint=None, best_metric=None))
    config = {
        "model_name": "test-model", "seed": 42, "lora_r": 16, "lora_alpha": 32,
        "lora_dropout": 0.05, "max_seq_length": 512, "num_train_epochs": 3,
        "learning_rate": 0.0002, "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2, "gradient_accumulation_steps": 8,
    }
    metadata = write_experiment_metadata(
        tmp_path / "output", config, data_dir, trainer,
        training_kwargs={
            "eval_strategy": "epoch", "save_strategy": "epoch",
            "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
            "greater_is_better": False, "save_total_limit": 2,
        },
        tokenizer_identifier="test-model",
        quantized=True,
        quantization_config={"load_in_4bit": True},
        runtime={"torch": "test"},
        start_time=1.0,
        end_time=2.5,
    )
    assert metadata["trainer_configuration"]["metric_for_best_model"] == "eval_loss"
    assert metadata["quantization"]["requested"] is True
    assert metadata["runtime"]["torch"] == "test"
    assert metadata["timing"]["elapsed_seconds"] == 1.5
