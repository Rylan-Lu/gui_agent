from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gui_agent.datasets.webarena import parse_webarena_task


def discover_webarena_files(path: Path) -> list[Path]:
    """Discover WebArena JSON sources from one file or a directory."""

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError("Input must be a JSON file")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError("No JSON task files found")

    return files


def _load_raw_tasks(path: Path) -> list[tuple[int | None, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return list(enumerate(data))

    return [(None, data)]


def audit_webarena_files(files: list[Path]) -> dict:
    """Parse WebArena task sources and return aggregate statistics.

    Both canonical collection files (for example ``test.raw.json``) and
    legacy one-task-per-file configurations are supported.
    """

    sites = Counter()
    eval_types = Counter()

    seen_ids: dict[int, str] = {}
    records: list[dict] = []
    errors: list[dict] = []
    reference_actions = 0
    raw_task_count = 0

    for path in files:
        try:
            raw_tasks = _load_raw_tasks(path)
        except Exception as exc:
            errors.append(
                {
                    "source_file": path.name,
                    "task_index": None,
                    "error": repr(exc),
                }
            )
            continue

        raw_task_count += len(raw_tasks)

        for task_index, raw_task in raw_tasks:
            try:
                task = parse_webarena_task(
                    raw_task,
                    source_file=str(path),
                )

                if task.task_id in seen_ids:
                    raise ValueError(
                        f"Duplicate task_id={task.task_id}; "
                        f"previous source: {seen_ids[task.task_id]}"
                    )

                location = path.name
                if task_index is not None:
                    location = f"{path.name}[{task_index}]"

                seen_ids[task.task_id] = location
                sites.update(task.sites)
                eval_types.update(task.eval_types)
                reference_actions += task.reference_action_count

                records.append(
                    {
                        "source_file": path.name,
                        "task_index": task_index,
                        "task_id": task.task_id,
                        "sites": list(task.sites),
                        "require_login": task.require_login,
                        "require_reset": task.require_reset,
                        "eval_types": list(task.eval_types),
                        "reference_action_count": task.reference_action_count,
                    }
                )

            except Exception as exc:
                errors.append(
                    {
                        "source_file": path.name,
                        "task_index": task_index,
                        "error": repr(exc),
                    }
                )

    return {
        "total_files": len(files),
        "raw_tasks": raw_task_count,
        "parsed_tasks": len(records),
        "failed_tasks": len(errors),
        # Backward-compatible alias used by the earlier audit report/tests.
        "failed_files": len(errors),
        "site_distribution": dict(sites),
        "evaluation_distribution": dict(eval_types),
        "reference_actions": reference_actions,
        "records": records,
        "errors": errors,
        "status": "PASS" if not errors else "FAILED",
    }
