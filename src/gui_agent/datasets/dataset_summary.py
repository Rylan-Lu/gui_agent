from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DATASET_MANIFESTS = {
    "screenagent": Path("screenagent") / "manifest.json",
    "mind2web": Path("mind2web") / "manifest.json",
    "webarena": Path("webarena") / "manifest.json",
}


class DatasetSummaryError(RuntimeError):
    """Raised when processed dataset manifests are missing or inconsistent."""


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetSummaryError(f"Missing dataset manifest: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetSummaryError(f"Failed to read manifest {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise DatasetSummaryError(f"Manifest must contain a JSON object: {path}")

    return data


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise DatasetSummaryError(f"{context}.{key} must be an object")
    return value


def _require_nonempty_str(
    mapping: dict[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetSummaryError(f"{context}.{key} must be a non-empty string")
    return value


def _require_exact_int(
    mapping: dict[str, Any],
    key: str,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < minimum:
        raise DatasetSummaryError(
            f"{context}.{key} must be an integer >= {minimum}"
        )
    return value


def _validate_common_manifest(
    manifest: dict[str, Any],
    expected_dataset: str,
) -> tuple[int, dict[str, Any]]:
    dataset = _require_nonempty_str(manifest, "dataset", context=expected_dataset)
    if dataset != expected_dataset:
        raise DatasetSummaryError(
            f"Expected dataset {expected_dataset!r}, got {dataset!r}"
        )

    schema_version = _require_exact_int(
        manifest,
        "schema_version",
        context=expected_dataset,
        minimum=1,
    )

    source = _require_mapping(manifest, "source", context=expected_dataset)
    _require_nonempty_str(source, "revision", context=f"{expected_dataset}.source")
    _require_nonempty_str(source, "split", context=f"{expected_dataset}.source")

    quality = _require_mapping(manifest, "quality", context=expected_dataset)
    _require_exact_int(
        quality,
        "parse_failures",
        context=f"{expected_dataset}.quality",
        minimum=0,
    )

    return schema_version, source


def _screenagent_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    schema_version, source = _validate_common_manifest(manifest, "screenagent")
    raw = _require_mapping(manifest, "raw", context="screenagent")
    records = _require_mapping(manifest, "records", context="screenagent")
    quality = _require_mapping(manifest, "quality", context="screenagent")
    actions = _require_mapping(
        manifest, "action_distribution", context="screenagent"
    )

    sessions = _require_exact_int(raw, "sessions", context="screenagent.raw")
    json_files = _require_exact_int(raw, "json_files", context="screenagent.raw")
    images = _require_exact_int(raw, "images", context="screenagent.raw")
    total = _require_exact_int(records, "total", context="screenagent.records")
    normal = _require_exact_int(records, "normal", context="screenagent.records")
    negative = _require_exact_int(records, "negative", context="screenagent.records")
    exported = total

    if normal + negative != total:
        raise DatasetSummaryError(
            "screenagent record counts do not close: normal + negative != total"
        )

    if _require_exact_int(
        quality,
        "parse_failures",
        context="screenagent.quality",
    ) != 0:
        raise DatasetSummaryError("screenagent manifest reports parse failures")

    return {
        "schema_version": schema_version,
        "source": {
            "revision": source["revision"],
            "split": source["split"],
        },
        "inventory": {
            "sessions": sessions,
            "json_files": json_files,
            "images": images,
            "records": exported,
            "normal_records": normal,
            "negative_records": negative,
        },
        "action_distribution": dict(sorted(actions.items())),
        "quality": {
            "parse_failures": quality["parse_failures"],
            "unknown_actions": quality.get("unknown_actions", 0),
        },
        "artifact": "screenagent/manifest.json",
    }


def _mind2web_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    schema_version, source = _validate_common_manifest(manifest, "mind2web")
    raw = _require_mapping(manifest, "raw", context="mind2web")
    quality = _require_mapping(manifest, "quality", context="mind2web")
    actions = _require_mapping(manifest, "action_distribution", context="mind2web")
    candidates = _require_mapping(manifest, "candidates", context="mind2web")

    shards = _require_exact_int(raw, "shards", context="mind2web.raw")
    tasks = _require_exact_int(raw, "tasks", context="mind2web.raw")
    action_count = _require_exact_int(raw, "actions", context="mind2web.raw")
    unique_task_ids = _require_exact_int(
        raw, "unique_task_ids", context="mind2web.raw"
    )
    unique_action_uids = _require_exact_int(
        raw, "unique_action_uids", context="mind2web.raw"
    )
    exported_actions = _require_exact_int(
        quality, "exported_actions", context="mind2web.quality"
    )

    if tasks != unique_task_ids:
        raise DatasetSummaryError("mind2web task IDs are not unique")
    if action_count != unique_action_uids:
        raise DatasetSummaryError("mind2web action UIDs are not unique")
    if action_count != exported_actions:
        raise DatasetSummaryError("mind2web exported action count does not match raw")
    if quality["parse_failures"] != 0:
        raise DatasetSummaryError("mind2web manifest reports parse failures")

    with_positive = _require_exact_int(
        candidates,
        "actions_with_positive_candidates",
        context="mind2web.candidates",
    )
    without_positive = _require_exact_int(
        candidates,
        "actions_without_positive_candidates",
        context="mind2web.candidates",
    )
    if with_positive + without_positive != action_count:
        raise DatasetSummaryError(
            "mind2web candidate coverage does not close to action count"
        )

    return {
        "schema_version": schema_version,
        "source": {
            "revision": source["revision"],
            "split": source["split"],
        },
        "inventory": {
            "shards": shards,
            "tasks": tasks,
            "actions": action_count,
            "actions_with_positive_candidates": with_positive,
            "actions_without_positive_candidates": without_positive,
        },
        "action_distribution": dict(sorted(actions.items())),
        "quality": {
            "parse_failures": quality["parse_failures"],
            "duplicate_task_ids": quality.get("duplicate_task_ids", 0),
            "duplicate_action_uids": quality.get("duplicate_action_uids", 0),
        },
        "artifact": "mind2web/manifest.json",
    }


def _webarena_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    schema_version, source = _validate_common_manifest(manifest, "webarena")
    raw = _require_mapping(manifest, "raw", context="webarena")
    quality = _require_mapping(manifest, "quality", context="webarena")
    evaluation = _require_mapping(manifest, "evaluation", context="webarena")
    eval_distribution = _require_mapping(
        evaluation,
        "eval_type_distribution",
        context="webarena.evaluation",
    )
    sites = _require_mapping(manifest, "site_distribution", context="webarena")

    tasks = _require_exact_int(raw, "tasks", context="webarena.raw")
    unique_task_ids = _require_exact_int(
        raw, "unique_task_ids", context="webarena.raw"
    )
    task_id_min = _require_exact_int(raw, "task_id_min", context="webarena.raw")
    task_id_max = _require_exact_int(raw, "task_id_max", context="webarena.raw")
    exported_tasks = _require_exact_int(
        quality, "exported_tasks", context="webarena.quality"
    )

    if tasks != unique_task_ids:
        raise DatasetSummaryError("webarena task IDs are not unique")
    if tasks != exported_tasks:
        raise DatasetSummaryError("webarena exported task count does not match raw")
    if quality["parse_failures"] != 0:
        raise DatasetSummaryError("webarena manifest reports parse failures")

    return {
        "schema_version": schema_version,
        "source": {
            "revision": source["revision"],
            "split": source["split"],
        },
        "inventory": {
            "tasks": tasks,
            "task_id_min": task_id_min,
            "task_id_max": task_id_max,
        },
        "site_distribution": dict(sorted(sites.items())),
        "evaluation_type_distribution": dict(sorted(eval_distribution.items())),
        "quality": {
            "parse_failures": quality["parse_failures"],
            "duplicate_task_ids": quality.get("duplicate_task_ids", 0),
        },
        "artifact": "webarena/manifest.json",
    }


def build_dataset_summary(processed_root: str | Path) -> dict[str, Any]:
    """Build a deterministic project-level summary from validated manifests."""

    root = Path(processed_root)

    manifests = {
        dataset: _load_json_object(root / relative_path)
        for dataset, relative_path in DATASET_MANIFESTS.items()
    }

    return {
        "schema_version": 1,
        "datasets": {
            "screenagent": _screenagent_summary(manifests["screenagent"]),
            "mind2web": _mind2web_summary(manifests["mind2web"]),
            "webarena": _webarena_summary(manifests["webarena"]),
        },
        "notes": {
            "aggregation_policy": (
                "No cross-dataset total is reported because ScreenAgent records, "
                "Mind2Web actions, and WebArena tasks are different semantic units."
            )
        },
    }


def write_dataset_summary(
    processed_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write dataset_summary.json."""

    root = Path(processed_root)
    summary = build_dataset_summary(root)

    destination = (
        Path(output_path)
        if output_path is not None
        else root / "dataset_summary.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_name(destination.name + ".tmp")
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    try:
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return summary
