"""Fine-tune Qwen with LoRA using TRL's SFTTrainer."""

import argparse
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_training_kwargs(
    config: dict,
    training_signature: dict,
    bf16_supported: bool,
    cuda_available: bool,
    output_dir: str,
) -> dict:
    """Build version-compatible TrainingArguments/SFTConfig keyword arguments."""
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=5,
        seed=config["seed"],
        report_to="none",
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.get("greater_is_better", False),
        save_total_limit=config.get("save_total_limit", 2),
    )
    evaluation_strategy = config.get("eval_strategy", "epoch")
    if "eval_strategy" in training_signature:
        kwargs["eval_strategy"] = evaluation_strategy
    elif "evaluation_strategy" in training_signature:
        kwargs["evaluation_strategy"] = evaluation_strategy
    else:
        raise RuntimeError("Installed Transformers does not support evaluation strategy configuration.")
    if kwargs["save_strategy"] != evaluation_strategy:
        raise ValueError("save_strategy must match eval_strategy when selecting the best checkpoint.")
    for name, value in (("bf16", bf16_supported), ("fp16", bool(cuda_available and not bf16_supported))):
        if name in training_signature:
            kwargs[name] = value
    required = ("load_best_model_at_end", "metric_for_best_model", "greater_is_better")
    missing = [name for name in required if name not in training_signature]
    if missing:
        raise RuntimeError(
            "Installed Transformers does not support best-checkpoint selection fields: "
            + ", ".join(missing)
        )
    return kwargs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_data(data_dir: Path) -> dict:
    """Verify prepared splits before any dataset or model loading."""
    from src.data_validation import load_records, validate_records

    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest was not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset manifest is not valid JSON: {manifest_path}") from exc
    required_splits = ("train", "validation", "test")
    split_hashes = manifest.get("split_sha256")
    split_ids = manifest.get("ids")
    counts = manifest.get("counts")
    if not isinstance(split_hashes, dict) or not isinstance(split_ids, dict) or not isinstance(counts, dict):
        raise ValueError("Dataset manifest must contain split_sha256, ids, and counts.")
    if set(split_hashes) != set(required_splits):
        raise ValueError("Dataset manifest must hash train, validation, and test splits.")

    loaded: dict[str, list[dict]] = {}
    all_ids: list[str] = []
    for split in required_splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Prepared split was not found: {path}")
        actual_hash = _sha256(path)
        if actual_hash != split_hashes[split]:
            raise ValueError(f"{split}.jsonl SHA-256 does not match split_manifest.json.")
        records = load_records(path)
        report = validate_records(records, str(path))
        if report.missing_required_columns or report.invalid_rows:
            raise ValueError(f"{split}.jsonl is invalid: {report.to_dict()}")
        expected = [str(item) for item in split_ids.get(split, [])]
        actual = [str(record.get("id")) for record in records]
        if len(actual) != len(set(actual)):
            raise ValueError(f"{split}.jsonl contains duplicate IDs.")
        if actual != expected or len(actual) != counts.get(split):
            raise ValueError(f"{split}.jsonl IDs or record count do not match split_manifest.json.")
        loaded[split] = records
        all_ids.extend(actual)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Prepared split IDs overlap across train, validation, and test.")

    source = manifest.get("dataset", {})
    source_status = "unavailable"
    source_path = source.get("path")
    if source_path:
        candidate = Path(source_path)
        if not candidate.exists():
            candidate = Path.cwd() / source_path
        if candidate.exists():
            source_status = "verified" if _sha256(candidate) == source.get("sha256") else "mismatch"
            if source_status == "mismatch":
                raise ValueError("Canonical source dataset SHA-256 does not match split_manifest.json.")
    manifest["_source_verification"] = source_status
    return manifest


def validate_output_and_resume(
    output_dir: Path,
    resume_from_checkpoint: str | None,
    allow_existing_output: bool,
) -> None:
    """Prevent accidental mixing of runs and validate explicit resume paths."""
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint)
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError(
                "resume-from-checkpoint must be a directory containing trainer_state.json."
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not allow_existing_output:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use a new directory or "
                "pass --allow-existing-output explicitly."
            )
        if (output_dir / "experiment_metadata.json").exists() and not resume_from_checkpoint:
            raise FileExistsError(
                f"Existing experiment metadata found in {output_dir}; use a new output "
                "directory or explicitly resume from a checkpoint."
            )


def build_training_data_files(data_dir: Path) -> dict[str, str]:
    """Return only train and validation files; test is evaluation-only."""
    return {
        split: str(data_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }


def write_experiment_metadata(
    output_dir: Path,
    config: dict,
    data_dir: Path,
    trainer: object,
    training_kwargs: dict | None = None,
    tokenizer_identifier: str | None = None,
    quantized: bool = False,
    quantization_config: dict | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    runtime: dict | None = None,
) -> dict:
    """Persist manifest-linked training settings and realized checkpoint results."""
    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest was not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("dataset", {})
    source_path = Path(source["path"]) if source.get("path") else None
    if source_path is not None and not source_path.exists():
        source_path = Path.cwd() / source_path
    if source_path is None or not source_path.exists():
        source_verification = "unavailable"
    elif _sha256(source_path) == source.get("sha256"):
        source_verification = "verified"
    else:
        source_verification = "mismatch"
    state = trainer.state
    metadata = {
        "dataset": {
            "source": manifest.get("dataset"),
            "source_verification": source_verification,
            "split_sha256": manifest.get("split_sha256"),
            "train_ids": manifest.get("ids", {}).get("train", []),
            "validation_ids": manifest.get("ids", {}).get("validation", []),
            "train_record_count": len(manifest.get("ids", {}).get("train", [])),
            "validation_record_count": len(manifest.get("ids", {}).get("validation", [])),
        },
        "model_name": config["model_name"],
        "tokenizer_identifier": tokenizer_identifier or config["model_name"],
        "seed": config["seed"],
        "lora": {
            "r": config["lora_r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            key: config[key]
            for key in (
                "max_seq_length",
                "num_train_epochs",
                "learning_rate",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
            )
        },
        "trainer_configuration": {
            key: (training_kwargs or {}).get(key)
            for key in (
                "eval_strategy",
                "evaluation_strategy",
                "save_strategy",
                "load_best_model_at_end",
                "metric_for_best_model",
                "greater_is_better",
                "save_total_limit",
            )
        },
        "quantization": {
            "requested": quantized,
            "configuration": quantization_config,
        },
        "runtime": runtime or {},
        "output_directory": str(output_dir),
        "timing": {
            "started_at_epoch_seconds": start_time,
            "finished_at_epoch_seconds": end_time,
            "elapsed_seconds": end_time - start_time if start_time is not None and end_time is not None else None,
        },
        "checkpoint_selection": {
            "evaluation_split": "validation",
            "metric": "eval_loss",
            "direction": "minimize",
            "selected_checkpoint": state.best_model_checkpoint,
            "best_validation_loss": state.best_metric,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install requirements.txt.") from exc
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install requirements.txt.") from exc
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or config["output_dir"])
    validate_output_and_resume(output_dir, args.resume_from_checkpoint, args.allow_existing_output)
    validate_training_data(Path(args.data_dir))
    if not torch.cuda.is_available():
        print("Warning: CUDA is unavailable. Training on CPU will be very slow; use Colab for practical training.")
        if args.quantized:
            raise SystemExit(
                "QLoRA (--quantized) requires CUDA. Run this command in Google Colab with a GPU, "
                "or omit --quantized for a slow CPU smoke run."
            )
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install requirements.txt.") from exc

    bf16_supported = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    data_files = build_training_data_files(Path(args.data_dir))
    dataset = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    quantization_config = None
    if args.quantized:
        from transformers import BitsAndBytesConfig
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": str(compute_dtype),
        }
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **model_kwargs)
    lora = LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    training_signature = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = build_training_kwargs(
        config,
        training_signature,
        bf16_supported,
        torch.cuda.is_available(),
        output_dir,
    )
    try:
        from trl import SFTConfig
    except ImportError:
        training_args = TrainingArguments(**training_kwargs)
    else:
        training_kwargs.update({
            "dataset_text_field": "text",
            "max_length": config["max_seq_length"],
        })
        training_args = SFTConfig(**training_kwargs)
    trainer_signature = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model, args=training_args, train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], peft_config=lora,
    )
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_signature:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_signature:
        trainer_kwargs["max_seq_length"] = config["max_seq_length"]
    trainer = SFTTrainer(**trainer_kwargs)
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    end_time = time.time()
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "trl": __import__("trl").__version__,
        "peft": __import__("peft").__version__,
        "accelerate": __import__("accelerate").__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    Path(output_dir, "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = write_experiment_metadata(
        output_dir, config, Path(args.data_dir), trainer, training_kwargs,
        config["model_name"], args.quantized, quantization_config,
        start_time, end_time, runtime,
    )
    print(
        f"Saved LoRA adapter to {output_dir}; selected checkpoint: "
        f"{metadata['checkpoint_selection']['selected_checkpoint']}; "
        f"best validation loss: {metadata['checkpoint_selection']['best_validation_loss']}"
    )


if __name__ == "__main__":
    main()
