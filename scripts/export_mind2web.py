from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from gui_agent.datasets.mind2web import parse_mind2web_task


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT.parent / "Mind2Web_reference"

DEFAULT_INPUT = DATA_ROOT / "data" / "train"

DEFAULT_OUTPUT = ROOT / "data" / "processed" / "mind2web"


def shard_sort_key(path: Path) -> int:
    """Sort train_2.json before train_10.json."""

    match = re.fullmatch(r"train_(\d+)\.json", path.name)

    if match is None:
        raise ValueError(f"Invalid shard filename: {path.name}")

    return int(match.group(1))


def discover_shards(
    input_path: Path,
    selected: list[str] | None = None,
) -> list[Path]:
    """Discover existing shards without downloading data."""

    if input_path.is_file():
        if selected:
            raise ValueError(
                "--shards cannot be used when --input is a file"
            )

        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    if selected:
        shards = []

        for name in selected:
            if re.fullmatch(r"train_\d+\.json", name) is None:
                raise ValueError(f"Invalid shard name: {name}")

            path = input_path / name

            if not path.is_file():
                raise FileNotFoundError(path)

            shards.append(path)

        if len(set(shards)) != len(shards):
            raise ValueError("Duplicate shard names")

    else:
        shards = list(input_path.glob("train_*.json"))

    if not shards:
        raise ValueError("No training shards found")

    return sorted(shards, key=shard_sort_key)


def portable_source_path(
    source: Path,
    source_root: Path,
) -> str:
    """Avoid writing machine-specific absolute paths."""

    try:
        return source.resolve().relative_to(
            source_root.resolve()
        ).as_posix()

    except ValueError:
        return source.name


def export_shard(
    source: Path,
    output_dir: Path,
    source_root: Path = DATA_ROOT,
) -> dict:
    """Export one shard and verify it before publishing."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{source.stem}.jsonl"

    stats_file = output_dir / f"{source.stem}_stats.json"

    temp_file = output_dir / f"{source.stem}.jsonl.tmp"

    source_name = portable_source_path(
        source,
        source_root,
    )

    # Load only the current shard.
    with source.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("Expected a JSON list of tasks")

    raw_actions = 0
    exported_examples = 0
    action_counts = Counter()
    seen_task_ids = set()

    try:
        with temp_file.open("w", encoding="utf-8") as f:

            for task_index, task in enumerate(tasks):

                if not isinstance(task, dict):
                    raise ValueError(
                        f"Task {task_index} must be an object"
                    )

                actions = task.get("actions")

                if not isinstance(actions, list):
                    raise ValueError(
                        f"Task {task_index} has invalid actions"
                    )

                task_id = task.get("annotation_id")

                if task_id in seen_task_ids:
                    raise ValueError(
                        f"Duplicate task ID: {task_id}"
                    )

                seen_task_ids.add(task_id)

                examples = parse_mind2web_task(
                    task,
                    source_file=source_name,
                )

                if len(examples) != len(actions):
                    raise RuntimeError(
                        f"Action count mismatch in task {task_index}"
                    )

                raw_actions += len(actions)

                for example in examples:

                    record = asdict(example)

                    # Make paths portable.
                    record["metadata"]["source_file"] = source_name

                    action_type = example.action.action_type.value

                    if action_type == "other":
                        raise RuntimeError(
                            f"Unknown action in task {task_index}"
                        )

                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )

                    action_counts[action_type] += 1
                    exported_examples += 1

                # Do not retain parsed examples from previous tasks.

        if raw_actions != exported_examples:
            raise RuntimeError(
                "Raw/exported action count mismatch"
            )

        # Validate the temporary JSONL line by line.
        verified_count = 0
        verified_actions = Counter()

        with temp_file.open("r", encoding="utf-8") as f:

            for line in f:

                record = json.loads(line)

                verified_count += 1

                verified_actions[
                    record["action"]["action_type"]
                ] += 1

        if verified_count != exported_examples:
            raise RuntimeError(
                "Exported line count mismatch"
            )

        if verified_actions != action_counts:
            raise RuntimeError(
                "Exported action distribution mismatch"
            )

        # Publish only after all validation succeeds.
        os.replace(temp_file, output_file)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    stats = {
        "source": "mind2web",
        "source_file": source_name,
        "total_tasks": len(tasks),
        "raw_actions": raw_actions,
        "exported_examples": exported_examples,
        "action_distribution": dict(action_counts),
        "unknown_actions": action_counts["other"],
        "output_file": output_file.name,
        "status": "PASS",
    }

    stats_temp = stats_file.with_suffix(".json.tmp")

    try:
        stats_temp.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        os.replace(stats_temp, stats_file)

    finally:
        if stats_temp.exists():
            stats_temp.unlink()

    return stats


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Export Mind2Web training shards."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input training directory or one JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--shards",
        nargs="+",
        help="Optional shard filenames, e.g. train_0.json.",
    )

    args = parser.parse_args()

    shards = discover_shards(
        args.input,
        args.shards,
    )

    print("===== Mind2Web Batch Export =====")
    print("Discovered shards:", len(shards))

    results = []
    errors = []

    for index, source in enumerate(shards, start=1):

        print(
            f"\n[{index}/{len(shards)}] {source.name}"
        )

        try:
            stats = export_shard(
                source,
                args.output_dir,
            )

            results.append(stats)

            print(
                "PASS:",
                stats["exported_examples"],
                "examples",
            )

        except Exception as exc:

            error = {
                "source_file": source.name,
                "error": repr(exc),
            }

            errors.append(error)

            print("FAILED:", repr(exc))

    summary = {
        "total_shards": len(shards),
        "successful_shards": len(results),
        "failed_shards": len(errors),
        "total_tasks": sum(
            item["total_tasks"]
            for item in results
        ),
        "total_actions": sum(
            item["exported_examples"]
            for item in results
        ),
        "results": results,
        "errors": errors,
        "status": "PASS" if not errors else "FAILED",
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = args.output_dir / "_batch_stats.json"

    report_temp = args.output_dir / "_batch_stats.json.tmp"

    try:
        report_temp.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(report_temp, report_file)

    finally:
        if report_temp.exists():
            report_temp.unlink()

    print("\n===== Batch Summary =====")

    print("Successful shards:", len(results))
    print("Failed shards:", len(errors))
    print("Total tasks:", summary["total_tasks"])
    print("Total actions:", summary["total_actions"])

    print("Report:", report_file)

    if errors:
        print("\nBATCH EXPORT: FAILED")
        raise SystemExit(1)

    print("\nBATCH EXPORT: PASS")


if __name__ == "__main__":
    main()