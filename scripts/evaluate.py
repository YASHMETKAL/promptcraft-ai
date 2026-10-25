"""Generate paired baseline/fine-tuned outputs for the held-out test set."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import generate_optimized_prompt
from src.model_loader import load_model_and_tokenizer
from src.evaluation import evaluate_three_systems, generate_prompt_engineering_baseline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_held_out_test(path: Path) -> None:
    if path.name != "test.jsonl":
        raise SystemExit(
            "Evaluation must use the held-out test split: pass a path ending in `test.jsonl`."
        )
    manifest_path = path.parent / "split_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing split manifest next to `{path}`. Prepare the dataset before evaluation."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = manifest.get("ids", {}).get("test")
    if not isinstance(expected_ids, list) or len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("The split manifest has no valid unique held-out test ID list.")
    actual_ids = [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(actual_ids) != len(set(actual_ids)):
        raise SystemExit("The supplied test.jsonl contains duplicate IDs.")
    if actual_ids != expected_ids:
        raise SystemExit(
            "The supplied test.jsonl IDs do not match the held-out test IDs in split_manifest.json."
        )
    expected_hash = manifest.get("split_sha256", {}).get("test")
    if not expected_hash:
        raise SystemExit(
            "The split manifest has no SHA-256 hash for test.jsonl; regenerate preparation outputs "
            "before evaluation."
        )
    if _sha256(path) != expected_hash:
        raise SystemExit("The supplied test.jsonl hash does not match split_manifest.json.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter", default="models/promptcraft-lora")
    parser.add_argument("--output", default="outputs/evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _validate_held_out_test(Path(args.test))
    model_name = args.model_name
    if model_name is None:
        import yaml
        model_name = yaml.safe_load(open("config/training_config.yaml", encoding="utf-8"))["model_name"]
    tuned, tuned_tokenizer, _ = load_model_and_tokenizer(model_name, args.adapter)
    tuned_fn = lambda prompt, **context: generate_optimized_prompt(
        tuned, tuned_tokenizer, prompt, **context
    )
    rows = evaluate_three_systems(
        args.test,
        generate_prompt_engineering_baseline,
        tuned_fn,
        args.output,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {len(rows)} three-way results and blank rating templates "
        f"alongside {args.output}. Add human rubric scores before interpreting quality."
    )


if __name__ == "__main__":
    main()
