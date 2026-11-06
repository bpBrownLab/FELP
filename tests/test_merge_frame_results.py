"""Source and frame identifiers remain unique when merging score batches."""
import csv
import json
from pathlib import Path


def test_merge_assigns_global_source_ids_and_preserves_local_ids(
    tmp_path: Path,
) -> None:
    from scripts.merge_frame_results import merge_frame_results

    header = ["frame_index", "trajectory_id", "metadata"]
    parts: list[Path] = []
    for name, local_ids in (("common_wt", [0, 0]), ("seed_01", [0, 1])):
        part = tmp_path / name
        part.mkdir()
        parts.append(part)
        with (part / "frame_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for index, local_id in enumerate(local_ids):
                writer.writerow(
                    {
                        "frame_index": index,
                        "trajectory_id": local_id,
                        "metadata": json.dumps({"kept": True}),
                    }
                )

    output = tmp_path / "merged.csv"
    assert merge_frame_results(parts, output) == 4
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["trajectory_id"]) for row in rows] == [0, 0, 1, 2]
    metadata = [json.loads(row["metadata"]) for row in rows]
    assert [row["source_group"] for row in metadata] == [
        "common_wt",
        "common_wt",
        "seed_01",
        "seed_01",
    ]
    assert [row["source_trajectory_id"] for row in metadata] == [0, 0, 0, 1]
