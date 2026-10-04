from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gui_agent.datasets.mind2web import parse_mind2web_task


SCHEMA_VERSION = 1
SOURCE_REPOSITORY = "https://huggingface.co/datasets/osunlp/Mind2Web"


def shard_sort_key(path: Path) -> int:
    match = re.fullmatch(r"train_(\d+)\.json", path.name)
    if match is None:
        raise ValueError(f"Invalid Mind2Web shard filename: {path.name}")
    return int(match.group(1))


def discover_mind2web_shards(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)

    if input_path.is_file():
        shard_sort_key(input_path)
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    shards = sorted(
        input_path.glob("train_*.json"),
        key=shard_sort_key,
    )
    if not shards:
        raise ValueError("No Mind2Web train_*.json shards found")

    shard_indices = [shard_sort_key(path) for path in shards]
    if len(shard_indices) != len(set(shard_indices)):
        raise ValueError("Duplicate Mind2Web shard indices")

    return shards


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def export_mind2web_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    """Export all selected Mind2Web train shards to one compact JSONL.

    Large raw/cleaned HTML strings are validated but intentionally not copied
    into the processed JSONL. Each processed record keeps source_file,
    annotation_id/task_id, and action_uid so the raw source action remains
    addressable.
    """

    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source_revision must be a non-empty string")

    shards = discover_mind2web_shards(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "actions.jsonl"
    manifest_file = output_dir / "manifest.json"
    temp_file = output_dir / "actions.jsonl.tmp"

    seen_task_ids: set[str] = set()
    seen_action_uids: set[str] = set()

    operation_distribution = Counter()
    original_op_distribution = Counter()
    positive_candidate_count_distribution = Counter()
    original_target_count_distribution = Counter()

    stats = Counter()
    shard_stats: list[dict[str, Any]] = []

    try:
        with temp_file.open("w", encoding="utf-8") as out:
            for shard in shards:
                with shard.open("r", encoding="utf-8") as f:
                    tasks = json.load(f)

                if not isinstance(tasks, list):
                    raise ValueError(
                        f"{shard.name}: root must be a JSON list"
                    )

                shard_task_count = 0
                shard_action_count = 0

                for task_index, task in enumerate(tasks):
                    if not isinstance(task, dict):
                        raise ValueError(
                            f"{shard.name}: task {task_index} must be an object"
                        )

                    task_id = task.get("annotation_id")
                    if not isinstance(task_id, str) or not task_id.strip():
                        raise ValueError(
                            f"{shard.name}: task {task_index} has invalid annotation_id"
                        )
                    if task_id in seen_task_ids:
                        raise ValueError(
                            f"Duplicate annotation_id across shards: {task_id}"
                        )
                    seen_task_ids.add(task_id)

                    actions = task.get("actions")
                    if not isinstance(actions, list):
                        raise ValueError(
                            f"{shard.name}: task {task_index} has invalid actions"
                        )

                    examples = parse_mind2web_task(
                        task,
                        source_file=shard.name,
                    )
                    if len(examples) != len(actions):
                        raise RuntimeError(
                            f"{shard.name}: task {task_index} action count mismatch"
                        )

                    shard_task_count += 1
                    stats["tasks"] += 1

                    for action_index, example in enumerate(examples):
                        if example.action is None:
                            raise RuntimeError(
                                f"{shard.name}: task {task_index} action {action_index} missing action"
                            )

                        action_uid = example.metadata.get("action_uid")
                        if action_uid in seen_action_uids:
                            raise ValueError(
                                f"Duplicate action_uid across shards: {action_uid}"
                            )
                        seen_action_uids.add(action_uid)

                        metadata = example.action.metadata
                        pos_count = metadata["positive_candidate_count"]
                        neg_count = metadata["negative_candidate_count"]
                        original_count = metadata["original_target_count"]

                        stats["actions"] += 1
                        stats["positive_candidates"] += pos_count
                        stats["negative_candidates"] += neg_count
                        if pos_count == 0:
                            stats["actions_without_positive_candidates"] += 1
                        else:
                            stats["actions_with_positive_candidates"] += 1

                        operation_distribution[
                            example.action.action_type.value
                        ] += 1
                        original_op_distribution[
                            metadata["original_op"]
                        ] += 1
                        positive_candidate_count_distribution[
                            str(pos_count)
                        ] += 1
                        original_target_count_distribution[
                            str(original_count)
                        ] += 1

                        out.write(
                            json.dumps(
                                asdict(example),
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )

                        shard_action_count += 1

                shard_stats.append(
                    {
                        "file": shard.name,
                        "tasks": shard_task_count,
                        "actions": shard_action_count,
                    }
                )

        # Verify the complete temporary JSONL before publishing it.
        verified_count = 0
        verified_distribution = Counter()
        with temp_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid exported JSONL at line {line_no}"
                    ) from exc

                verified_count += 1
                verified_distribution[
                    record["action"]["action_type"]
                ] += 1

        if verified_count != stats["actions"]:
            raise RuntimeError("Exported action count mismatch")
        if verified_distribution != operation_distribution:
            raise RuntimeError("Exported action distribution mismatch")

        os.replace(temp_file, output_file)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    manifest: dict[str, Any] = {
        "dataset": "mind2web",
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision,
            "split": "train",
            "shards": [shard.name for shard in shards],
        },
        "raw": {
            "shards": len(shards),
            "tasks": stats["tasks"],
            "actions": stats["actions"],
            "unique_task_ids": len(seen_task_ids),
            "unique_action_uids": len(seen_action_uids),
        },
        "action_distribution": dict(sorted(operation_distribution.items())),
        "original_operation_distribution": dict(
            sorted(original_op_distribution.items())
        ),
        "candidates": {
            "positive_total": stats["positive_candidates"],
            "negative_total": stats["negative_candidates"],
            "actions_with_positive_candidates": stats[
                "actions_with_positive_candidates"
            ],
            "actions_without_positive_candidates": stats[
                "actions_without_positive_candidates"
            ],
            "positive_candidate_count_distribution": dict(
                sorted(
                    positive_candidate_count_distribution.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "original_target_count_distribution": dict(
                sorted(
                    original_target_count_distribution.items(),
                    key=lambda item: int(item[0]),
                )
            ),
        },
        "quality": {
            "parse_failures": 0,
            "duplicate_task_ids": 0,
            "duplicate_action_uids": 0,
            "exported_actions": verified_count,
        },
        "shard_stats": shard_stats,
        "files": {
            "actions": output_file.name,
        },
    }

    _atomic_write_json(manifest_file, manifest)
    return manifest
