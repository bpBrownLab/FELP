#!/usr/bin/env python3
"""Merge frame_results.csv files from separate scoring runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

csv.field_size_limit(sys.maxsize)


def merge_frame_results(part_dirs: list[Path], output_path: Path) -> int:
    """Merge score tables, renumbering trajectories and keeping original IDs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    frame_index = 0
    rows_written = 0
    next_trajectory_id = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for part_dir in part_dirs:
            trajectory_ids: dict[str, int] = {}
            source = part_dir / "frame_results.csv"
            with source.open(encoding="utf-8", newline="") as reader_handle:
                reader = csv.DictReader(reader_handle)
                if header is None:
                    header = list(reader.fieldnames or [])
                    writer = csv.DictWriter(handle, fieldnames=header)
                    writer.writeheader()
                assert writer is not None
                for row in reader:
                    local_trajectory_id = str(row.get("trajectory_id", "")).strip()
                    if local_trajectory_id not in trajectory_ids:
                        trajectory_ids[local_trajectory_id] = next_trajectory_id
                        next_trajectory_id += 1
                    metadata: dict[str, object]
                    try:
                        raw_metadata = json.loads(row.get("metadata") or "{}")
                        metadata = (
                            dict(raw_metadata)
                            if isinstance(raw_metadata, dict)
                            else {"legacy_metadata": raw_metadata}
                        )
                    except json.JSONDecodeError:
                        metadata = {"legacy_metadata": row.get("metadata", "")}
                    metadata.update(
                        {
                            "source_group": part_dir.name,
                            "source_trajectory_id": (
                                int(local_trajectory_id)
                                if local_trajectory_id.lstrip("-").isdigit()
                                else local_trajectory_id or None
                            ),
                        }
                    )
                    row["frame_index"] = str(frame_index)
                    row["trajectory_id"] = str(
                        trajectory_ids[local_trajectory_id]
                    )
                    row["metadata"] = json.dumps(metadata, sort_keys=True)
                    writer.writerow(row)
                    frame_index += 1
                    rows_written += 1
    return rows_written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part_dirs", type=Path, nargs="+", help="Directories containing frame_results.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    for directory in args.part_dirs:
        if not (directory / "frame_results.csv").is_file():
            parser.error(f"Missing frame_results.csv in {directory}")
    count = merge_frame_results(args.part_dirs, args.output)
    print(f"Wrote {count} frames to {args.output}")


if __name__ == "__main__":
    main()
