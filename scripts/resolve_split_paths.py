"""Resolve the original split CSV paths against a local dataset root."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def resolve_path(original_path: str, dataset_root: Path) -> Path:
    """Replace the original machine-specific prefix with a local dataset root."""
    parts = Path(original_path).parts
    try:
        data_index = next(index for index, part in enumerate(parts) if part.lower() == "data")
    except StopIteration as error:
        raise ValueError(f"Could not locate the data folder in path: {original_path}") from error

    return dataset_root.joinpath(*parts[data_index + 1 :])


def resolve_split(input_path: Path, output_path: Path, dataset_root: Path) -> int:
    """Write a portable copy of one split while preserving every original row."""
    with input_path.open("r", newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None or "filepath" not in reader.fieldnames:
            raise ValueError(f"Missing filepath column in {input_path}")

        rows = list(reader)

    missing = []
    for row in rows:
        local_path = resolve_path(row["filepath"], dataset_root)
        row["filepath"] = str(local_path)
        if not local_path.is_file():
            missing.append(local_path)

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image files are missing for {input_path}.\n{preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/portable"))
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    for split_name in ("train_split.csv", "val_split.csv", "test_split.csv"):
        count = resolve_split(
            args.input_dir / split_name,
            args.output_dir / split_name,
            dataset_root,
        )
        print(f"Resolved {split_name}: {count} rows")


if __name__ == "__main__":
    main()