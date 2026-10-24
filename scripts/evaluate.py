"""Generate paired baseline/fine-tuned outputs for the held-out test set."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import generate_optimized_prompt
from src.model_loader import load_model_and_tokenizer
from src.evaluation import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter", default="models/promptcraft-lora")
    parser.add_argument("--output", default="outputs/evaluation_results.csv")
    args = parser.parse_args()
    model_name = args.model_name
    if model_name is None:
        import yaml
        model_name = yaml.safe_load(open("config/training_config.yaml", encoding="utf-8"))["model_name"]
    baseline, tokenizer, _ = load_model_and_tokenizer(model_name)
    tuned, tuned_tokenizer, _ = load_model_and_tokenizer(model_name, args.adapter)
    baseline_fn = lambda prompt: generate_optimized_prompt(baseline, tokenizer, prompt)
    tuned_fn = lambda prompt: generate_optimized_prompt(tuned, tuned_tokenizer, prompt)
    rows = evaluate_models(args.test, baseline_fn, tuned_fn, args.output)
    print(f"Wrote {len(rows)} paired results to {args.output}. Add human rubric scores before interpreting quality.")


if __name__ == "__main__":
    main()
