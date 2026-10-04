from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from gui_agent.datasets.webarena import (
    WebArenaTask,
    load_webarena_collection,
)


SCHEMA_VERSION = 1
SOURCE_REPOSITORY = "https://github.com/web-arena-x/webarena"


def _task_to_record(task: WebArenaTask) -> dict[str, Any]:
    return {
        "source": "webarena",
        "task_id": task.task_id,
        "intent": task.intent,
        "sites": list(task.sites),
        "start_url": task.start_url,
        "require_login": task.require_login,
        "require_reset": task.require_reset,
        "storage_state": task.storage_state,
        "geolocation": task.geolocation,
        "intent_template": task.intent_template,
        "intent_template_id": task.intent_template_id,
        "instantiation_dict": task.instantiation_dict,
        "evaluation": task.evaluation,
        "reference_action_sequence": task.reference_action_sequence,
        "string_note": task.string_note,
        "source_file": task.source_file,
    }


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                file.write("\n")

        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    try:
        temp.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def export_webarena_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    """Export a canonical WebArena task collection to JSONL + manifest."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("source_revision must be a non-empty string")

    tasks = load_webarena_collection(input_path)

    task_ids: set[int] = set()
    duplicate_task_ids: list[int] = []

    sites = Counter()
    site_combinations = Counter()
    eval_types = Counter()
    reference_answer_keys = Counter()
    reference_answer_types = Counter()

    storage_state_present = 0
    storage_state_missing = 0
    top_level_string_notes = 0
    reference_action_tasks = 0
    reference_actions = 0

    records: list[dict[str, Any]] = []

    for task in tasks:
        if task.task_id in task_ids:
            duplicate_task_ids.append(task.task_id)
        else:
            task_ids.add(task.task_id)

        sites.update(task.sites)
        site_combinations["|".join(task.sites)] += 1
        eval_types.update(task.eval_types)

        reference_answers = task.evaluation.get("reference_answers")
        reference_answer_types[type(reference_answers).__name__] += 1
        if isinstance(reference_answers, dict):
            reference_answer_keys.update(reference_answers.keys())

        if task.storage_state is None:
            storage_state_missing += 1
        else:
            storage_state_present += 1

        if task.string_note is not None:
            top_level_string_notes += 1

        if task.reference_action_sequence is not None:
            reference_action_tasks += 1
            reference_actions += task.reference_action_count

        records.append(_task_to_record(task))

    if duplicate_task_ids:
        sample = duplicate_task_ids[:10]
        raise RuntimeError(f"Duplicate WebArena task_id values: {sample}")

    if len(task_ids) != len(tasks):
        raise RuntimeError("WebArena task identifier count mismatch")

    manifest: dict[str, Any] = {
        "dataset": "webarena",
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": source_revision.strip(),
            "split": "test",
            "file": input_path.name,
        },
        "raw": {
            "tasks": len(tasks),
            "unique_task_ids": len(task_ids),
            "task_id_min": min(task_ids) if task_ids else None,
            "task_id_max": max(task_ids) if task_ids else None,
        },
        "site_distribution": dict(sorted(sites.items())),
        "site_combination_distribution": dict(
            sorted(site_combinations.items())
        ),
        "evaluation": {
            "eval_type_distribution": dict(sorted(eval_types.items())),
            "reference_answer_type_distribution": dict(
                sorted(reference_answer_types.items())
            ),
            "reference_answer_key_distribution": dict(
                sorted(reference_answer_keys.items())
            ),
        },
        "environment": {
            "tasks_with_storage_state": storage_state_present,
            "tasks_without_storage_state": storage_state_missing,
            "top_level_string_note_tasks": top_level_string_notes,
        },
        "reference_actions": {
            "tasks_with_reference_action_sequence": reference_action_tasks,
            "total_reference_actions": reference_actions,
        },
        "quality": {
            "parse_failures": 0,
            "duplicate_task_ids": 0,
            "exported_tasks": len(records),
        },
        "files": {
            "tasks": "tasks.jsonl",
        },
    }

    tasks_path = output_dir / "tasks.jsonl"
    manifest_path = output_dir / "manifest.json"

    _atomic_write_jsonl(tasks_path, records)

    # Verify the exported artifact before publishing the manifest.
    exported_count = 0
    with tasks_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid exported JSONL at line {line_no}: {exc}"
                ) from exc
            if type(record.get("task_id")) is not int:
                raise RuntimeError(
                    f"Invalid exported task_id at line {line_no}"
                )
            exported_count += 1

    if exported_count != len(records):
        raise RuntimeError(
            "Exported WebArena task count does not match parsed task count"
        )

    _atomic_write_json(manifest_path, manifest)
    return manifest
