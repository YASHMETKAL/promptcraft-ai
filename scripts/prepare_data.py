"""Prepare validated train/validation/test JSONL files."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_preparation import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_dataset(args.input, args.output_dir, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
