"""Create a separate 100-row dataset without overwriting the original source."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="data/promptcraft_custom_dataset.csv")
    parser.add_argument("--additions", default="data/promptcraft_ai_assisted_additions.csv")
    parser.add_argument("--output", default="data/promptcraft_custom_dataset_100.csv")
    args = parser.parse_args()

    with Path(args.original).open(newline="", encoding="utf-8-sig") as handle:
        original = list(csv.DictReader(handle))
    with Path(args.additions).open(newline="", encoding="utf-8-sig") as handle:
        additions = list(csv.DictReader(handle))
    if len(original) != 82 or len(additions) != 18:
        raise ValueError(f"Expected 82 original and 18 additions, got {len(original)} and {len(additions)}.")
    ids = {row["id"] for row in original}
    if ids.intersection(row["id"] for row in additions):
        raise ValueError("Addition IDs overlap with the original dataset.")
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=original[0].keys())
        writer.writeheader()
        writer.writerows(original + additions)
    print(f"Wrote {len(original) + len(additions)} rows to {output}")


if __name__ == "__main__":
    main()
