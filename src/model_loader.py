"""Load a base model and an optional trained LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    adapter_path: str | None = None,
    quantized: bool = False,
) -> tuple[Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before loading the model.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs: dict[str, Any] = {"device_map": "auto" if torch.cuda.is_available() else None}
    if quantized:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        except ImportError as exc:
            raise RuntimeError("QLoRA requested but bitsandbytes is unavailable.") from exc
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    mode = "base model"
    if adapter_path:
        adapter = Path(adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f"Fine-tuned adapter was not found: {adapter}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install peft to load a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, adapter)
        mode = f"fine-tuned adapter ({adapter})"
    return model, tokenizer, mode
