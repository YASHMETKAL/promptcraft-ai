"""Fine-tune Qwen with LoRA using TRL's SFTTrainer."""

import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quantized", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install requirements.txt.") from exc
    if not torch.cuda.is_available():
        print("Warning: CUDA is unavailable. Training on CPU will be very slow; use Colab for practical training.")
        if args.quantized:
            raise SystemExit(
                "QLoRA (--quantized) requires CUDA. Run this command in Google Colab with a GPU, "
                "or omit --quantized for a slow CPU smoke run."
            )
    try:
        import yaml
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install requirements.txt.") from exc

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = args.output_dir or config["output_dir"]
    bf16_supported = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    data_files = {name: str(Path(args.data_dir) / f"{name}.jsonl") for name in ("train", "validation")}
    dataset = load_dataset("json", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if args.quantized:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **model_kwargs)
    lora = LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    training_kwargs = dict(
        output_dir=output_dir, num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"], per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        save_strategy="epoch", logging_steps=5, seed=config["seed"],
        report_to="none",
    )
    # Transformers 4.46 and later renamed this argument to eval_strategy.
    training_signature = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs["eval_strategy" if "eval_strategy" in training_signature else "evaluation_strategy"] = "epoch"
    if "bf16" in training_signature:
        training_kwargs["bf16"] = bf16_supported
    if "fp16" in training_signature:
        training_kwargs["fp16"] = bool(torch.cuda.is_available() and not bf16_supported)
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
    trainer.train(resume_from_checkpoint=args.output_dir if args.output_dir else None)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    Path(output_dir, "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
