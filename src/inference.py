"""Inference helpers with explicit model-mode reporting."""

from __future__ import annotations

import time
from typing import Any


def generate_optimized_prompt(
    model: Any,
    tokenizer: Any,
    original_prompt: str,
    category: str = "General",
    audience: str = "",
    output_format: str = "",
    tone: str = "",
    requirements: str = "",
    max_new_tokens: int = 220,
) -> tuple[str, float]:
    if not original_prompt.strip():
        raise ValueError("Original prompt cannot be empty.")
    user = (
        "You are a prompt optimizer, not an answer generator. Rewrite the request below "
        "as one standalone, reusable prompt for another AI assistant. Output only that "
        "rewritten prompt in plain text. Do not answer the request, do not create a quiz, "
        "do not add an answer key, and do not include headings, commentary, or markdown "
        "unless the user explicitly requested them. Preserve the original intent and add "
        "only useful constraints from the supplied fields.\n\n"
        f"Request: {original_prompt.strip()}\nCategory: {category}\n"
        f"Audience: {audience or 'not specified'}\nFormat: {output_format or 'not specified'}\n"
        f"Tone: {tone or 'not specified'}\nRequirements: {requirements or 'none'}"
    )
    messages = [{"role": "user", "content": user}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError):
        text = f"<|user|>\n{user}\n<|assistant|>\n"
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start
    generated = output[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not result:
        raise RuntimeError("The model returned an empty response.")
    return result, elapsed
