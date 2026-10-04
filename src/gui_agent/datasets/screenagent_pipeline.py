from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from gui_agent.datasets.schema import ActionType, GUIExample
from gui_agent.datasets.screenagent import load_screenagent_file


SCREENAGENT_DATASET_NAME = "screenagent"
SCREENAGENT_SCHEMA_VERSION = 1
SCREENAGENT_DEFAULT_REPOSITORY = "https://github.com/niuzaisheng/ScreenAgent"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _require_directory(path: str | Path, *, field_name: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{field_name} directory not found: {path}")
    return path


def _require_nonempty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _relative_posix(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside dataset root: {path}") from exc
    return relative.as_posix()


def serialize_example(example: GUIExample, *, data_root: str | Path) -> dict[str, Any]:
    """Convert GUIExample to a portable JSON-compatible record."""
    root = Path(data_root)
    record = asdict(example)

    screenshot = record.get("screenshot")
    if screenshot is not None:
        record["screenshot"] = _relative_posix(screenshot, root)

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("GUIExample metadata must serialize to an object")

    source_file = metadata.get("source_file")
    if source_file is not None:
        metadata["source_file"] = _relative_posix(source_file, root)

    return record


def _session_id_from_path(path: Path, data_root: Path) -> str:
    relative = path.relative_to(data_root)
    if not relative.parts:
        raise ValueError(f"cannot derive session id from path: {path}")
    return relative.parts[0]


def _count_images(data_root: Path) -> int:
    return sum(
        1
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _collect_source(
    data_root: str | Path,
    *,
    collect_records: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = _require_directory(data_root, field_name="data_root")
    files = sorted(root.rglob("*.json"))

    if not files:
        raise ValueError(
            f"no ScreenAgent JSON files found under: {root}"
        )

    stats = Counter()
    action_distribution = Counter()
    sessions: set[str] = set()
    failures: list[dict[str, str]] = []
    normal_records: list[dict[str, Any]] = []
    negative_records: list[dict[str, Any]] = []

    for path in files:
        stats["json_files"] += 1

        path_session_id = _session_id_from_path(
            path,
            root,
        )
        sessions.add(path_session_id)

        is_negative_file = (
            "neg_plan" in path.stem.lower()
        )
        stats[
            "negative_files"
            if is_negative_file
            else "normal_files"
        ] += 1

        # Validate the raw screenshot reference once per source JSON.
        # Empty-action files still need their image reference checked.
        try:
            raw_data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            stats["parse_failures"] += 1
            failures.append(
                {
                    "file": _relative_posix(path, root),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        if not isinstance(raw_data, dict):
            stats["parse_failures"] += 1
            failures.append(
                {
                    "file": _relative_posix(path, root),
                    "error_type": "ValueError",
                    "error": "ScreenAgent file must contain a JSON object",
                }
            )
            continue

        saved_image_name = raw_data.get(
            "saved_image_name"
        )

        if (
            not isinstance(saved_image_name, str)
            or not saved_image_name.strip()
        ):
            stats["missing_image_paths"] += 1
        else:
            image_path = (
                path.parent
                / "images"
                / saved_image_name
            )

            if not image_path.is_file():
                stats["missing_image_files"] += 1

        try:
            examples = load_screenagent_file(path)
        except Exception as exc:
            stats["parse_failures"] += 1
            failures.append(
                {
                    "file": _relative_posix(path, root),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        stats["parsed_files"] += 1

        if not examples:
            stats["empty_action_files"] += 1
            continue

        first_metadata = examples[0].metadata

        expected_negative = first_metadata.get(
            "is_negative_plan"
        )
        if expected_negative is not is_negative_file:
            stats["classification_mismatches"] += 1

        parsed_session_id = first_metadata.get(
            "session_id"
        )
        if parsed_session_id != path_session_id:
            stats["session_id_mismatches"] += 1

        first_screenshot = examples[0].screenshot

        for index, example in enumerate(examples):
            stats["records_total"] += 1

            if example.step_index != index:
                stats["step_index_errors"] += 1

            if len(example.history) != index:
                stats["history_errors"] += 1

            if example.screenshot != first_screenshot:
                stats["screenshot_mismatches"] += 1

            if example.action is None:
                stats["missing_actions"] += 1
                continue

            action_distribution[
                example.action.action_type.value
            ] += 1

            if (
                example.action.action_type
                == ActionType.OTHER
            ):
                stats["unknown_actions"] += 1

            if is_negative_file:
                stats["negative_records"] += 1
            else:
                stats["normal_records"] += 1

            if collect_records:
                record = serialize_example(
                    example,
                    data_root=root,
                )

                if is_negative_file:
                    negative_records.append(record)
                else:
                    normal_records.append(record)

    audit = {
        "dataset": SCREENAGENT_DATASET_NAME,
        "schema_version": SCREENAGENT_SCHEMA_VERSION,
        "raw": {
            "sessions": len(sessions),
            "session_ids": sorted(sessions),
            "json_files": stats["json_files"],
            "images": _count_images(root),
            "normal_files": stats["normal_files"],
            "negative_files": stats["negative_files"],
            "empty_action_files": stats["empty_action_files"],
        },
        "records": {
            "total": stats["records_total"],
            "normal": stats["normal_records"],
            "negative": stats["negative_records"],
        },
        "action_distribution": dict(
            sorted(action_distribution.items())
        ),
        "quality": {
            "parsed_files": stats["parsed_files"],
            "parse_failures": stats["parse_failures"],
            "classification_mismatches": stats[
                "classification_mismatches"
            ],
            "session_id_mismatches": stats[
                "session_id_mismatches"
            ],
            "screenshot_mismatches": stats[
                "screenshot_mismatches"
            ],
            "unknown_actions": stats["unknown_actions"],
            "missing_actions": stats["missing_actions"],
            "missing_image_paths": stats[
                "missing_image_paths"
            ],
            "missing_image_files": stats[
                "missing_image_files"
            ],
            "step_index_errors": stats[
                "step_index_errors"
            ],
            "history_errors": stats["history_errors"],
        },
        "failures": failures,
    }

    return audit, normal_records, negative_records


def audit_screenagent_source(data_root: str | Path) -> dict[str, Any]:
    audit, _, _ = _collect_source(data_root, collect_records=False)
    return audit


def validate_source_audit(audit: dict[str, Any]) -> None:
    quality = audit.get("quality")

    if not isinstance(quality, dict):
        raise ValueError(
            "invalid ScreenAgent audit object"
        )

    required_zero = (
        "parse_failures",
        "classification_mismatches",
        "session_id_mismatches",
        "screenshot_mismatches",
        "missing_actions",
        "missing_image_paths",
        "missing_image_files",
        "step_index_errors",
        "history_errors",
    )

    problems = {
        key: int(quality.get(key, 0))
        for key in required_zero
        if int(quality.get(key, 0)) != 0
    }

    if problems:
        raise RuntimeError(
            "ScreenAgent source integrity check "
            f"failed: {problems}"
        )

    raw = audit.get("raw", {})
    records = audit.get("records", {})
    action_distribution = audit.get(
        "action_distribution",
        {},
    )

    if (
        raw.get("json_files")
        != quality.get("parsed_files")
    ):
        raise RuntimeError(
            "not every ScreenAgent JSON file "
            "parsed successfully"
        )

    if (
        records.get("total")
        != records.get("normal", 0)
        + records.get("negative", 0)
    ):
        raise RuntimeError(
            "ScreenAgent record totals "
            "are inconsistent"
        )

    if (
        sum(action_distribution.values())
        != records.get("total")
    ):
        raise RuntimeError(
            "ScreenAgent action distribution "
            "does not match total records"
        )


def _write_jsonl_file(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("JSONL record must be an object")
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def _validate_jsonl_file(path: Path, *, expected_count: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"empty JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
            count += 1
    if expected_count is not None and count != expected_count:
        raise RuntimeError(f"count mismatch for {path}: expected {expected_count}, got {count}")
    return count


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        written = _write_jsonl_file(temp_path, records)
        _validate_jsonl_file(temp_path, expected_count=written)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        parsed = json.loads(temp_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise RuntimeError(f"manifest did not serialize as an object: {path}")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def export_screenagent_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
    source_repository: str = SCREENAGENT_DEFAULT_REPOSITORY,
    source_split: str = "train",
) -> dict[str, Any]:
    revision = _require_nonempty_string(source_revision, field_name="source_revision")
    repository = _require_nonempty_string(source_repository, field_name="source_repository")
    split = _require_nonempty_string(source_split, field_name="source_split")

    root = _require_directory(data_root, field_name="data_root")
    out = Path(output_dir)

    audit, normal_records, negative_records = _collect_source(root, collect_records=True)
    validate_source_audit(audit)

    if len(normal_records) != audit["records"]["normal"]:
        raise RuntimeError("normal record collection count mismatch")
    if len(negative_records) != audit["records"]["negative"]:
        raise RuntimeError("negative record collection count mismatch")

    manifest = {
        "dataset": SCREENAGENT_DATASET_NAME,
        "schema_version": SCREENAGENT_SCHEMA_VERSION,
        "source": {
            "repository": repository,
            "revision": revision,
            "split": split,
        },
        "raw": audit["raw"],
        "records": audit["records"],
        "action_distribution": audit["action_distribution"],
        "quality": audit["quality"],
        "files": {
            "normal": "normal.jsonl",
            "negative": "negative.jsonl",
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(out / "normal.jsonl", normal_records)
    _atomic_write_jsonl(out / "negative.jsonl", negative_records)
    # Commit marker is written last.
    _atomic_write_json(out / "manifest.json", manifest)
    return manifest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file containing one JSON object per non-empty line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"empty line: {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"record must be an object: {path}:{line_number}")
            records.append(record)
    return records


def _load_screenagent_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = load_jsonl(path)
    for line_number, record in enumerate(records, start=1):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object: {path}:{line_number}")
        session_id = metadata.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"missing session_id: {path}:{line_number}")
    return records


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _get_session_id(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    return metadata["session_id"]


def split_screenagent_dataset(
    data_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    seed: int = 42,
    validation_ratio: float = 0.2,
) -> dict[str, Any]:
    data_dir = _require_directory(data_dir, field_name="data_dir")
    out = Path(output_dir) if output_dir is not None else data_dir / "splits"

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if isinstance(validation_ratio, bool) or not isinstance(validation_ratio, (int, float)):
        raise ValueError("validation_ratio must be a number")
    validation_ratio = float(validation_ratio)
    if not math.isfinite(validation_ratio) or not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be finite and between 0 and 1")

    source_manifest = _load_json_object(data_dir / "manifest.json")
    datasets = {
        "normal": _load_screenagent_records(data_dir / "normal.jsonl"),
        "negative": _load_screenagent_records(data_dir / "negative.jsonl"),
    }

    expected_records = source_manifest.get("records")
    if not isinstance(expected_records, dict):
        raise ValueError("source manifest is missing record counts")
    for name, records in datasets.items():
        expected = expected_records.get(name)
        if type(expected) is not int or expected < 0:
            raise ValueError(f"invalid source manifest count for {name}")
        if len(records) != expected:
            raise RuntimeError(
                f"{name} source count mismatch: manifest={expected}, actual={len(records)}"
            )

    all_sessions = sorted(
        {
            _get_session_id(record)
            for records in datasets.values()
            for record in records
        }
    )

    if len(all_sessions) < 2:
        raise ValueError(
            "at least two sessions are required for a split"
        )

    raw_manifest = source_manifest.get("raw")
    if not isinstance(raw_manifest, dict):
        raise ValueError(
            "source manifest is missing raw metadata"
        )

    source_session_ids = raw_manifest.get(
        "session_ids"
    )
    if not isinstance(source_session_ids, list):
        raise ValueError(
            "source manifest is missing session_ids"
        )

    source_session_set = set(source_session_ids)
    record_session_set = set(all_sessions)

    if not record_session_set.issubset(
        source_session_set
    ):
        raise RuntimeError(
            "processed records contain unknown "
            "ScreenAgent session ids"
        )

    shuffled = list(all_sessions)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * validation_ratio)))
    val_sessions = set(shuffled[:val_count])
    train_sessions = set(shuffled[val_count:])
    if train_sessions & val_sessions:
        raise RuntimeError("train/validation session sets overlap")

    splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    record_counts: dict[str, int] = {}
    for name, records in datasets.items():
        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        for record in records:
            if _get_session_id(record) in val_sessions:
                val_records.append(record)
            else:
                train_records.append(record)
        if len(train_records) + len(val_records) != len(records):
            raise RuntimeError(f"record loss detected while splitting {name}")
        splits[name] = {"train": train_records, "val": val_records}
        record_counts[f"{name}_train"] = len(train_records)
        record_counts[f"{name}_val"] = len(val_records)

    actual_train_sessions = {
        _get_session_id(record)
        for group in splits.values()
        for record in group["train"]
    }
    actual_val_sessions = {
        _get_session_id(record)
        for group in splits.values()
        for record in group["val"]
    }
    overlap = (
        actual_train_sessions
        & actual_val_sessions
    )

    if overlap:
        raise RuntimeError(
            "session leakage detected: "
            f"{sorted(overlap)}"
        )

    if actual_train_sessions != train_sessions:
        raise RuntimeError(
            "train session set changed while splitting"
        )

    if actual_val_sessions != val_sessions:
        raise RuntimeError(
            "validation session set changed "
            "while splitting"
        )

    if sum(record_counts.values()) != sum(len(records) for records in datasets.values()):
        raise RuntimeError("split total does not match source total")

    manifest = {
        "dataset": SCREENAGENT_DATASET_NAME,
        "schema_version": SCREENAGENT_SCHEMA_VERSION,
        "source_revision": source_manifest.get("source", {}).get("revision"),
        "source_split": source_manifest.get("source", {}).get("split"),
        "strategy": "session_level",
        "seed": seed,
        "validation_ratio": validation_ratio,
        "source_session_count": len(source_session_set),
        "session_count": len(all_sessions),
        "sessions_without_records": sorted(
            source_session_set - record_session_set
        ),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "train_session_ids": sorted(train_sessions),
        "val_session_ids": sorted(val_sessions),
        "session_overlap": 0,
        "record_counts": record_counts,
        "files": {
            "normal_train": "normal_train.jsonl",
            "normal_val": "normal_val.jsonl",
            "negative_train": "negative_train.jsonl",
            "negative_val": "negative_val.jsonl",
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    for name, group in splits.items():
        for split_name, records in group.items():
            _atomic_write_jsonl(out / f"{name}_{split_name}.jsonl", records)
    _atomic_write_json(out / "split_manifest.json", manifest)
    return manifest


def build_planner_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"planner source record {index} must be an object")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"planner source record {index} has invalid task_id")
        groups[task_id].append(record)

    samples: list[dict[str, Any]] = []
    for task_id, group in sorted(groups.items()):
        for record in group:
            if type(record.get("step_index")) is not int or record["step_index"] < 0:
                raise ValueError(f"invalid step_index in task {task_id}")
        group.sort(key=lambda item: item["step_index"])
        indices = [record["step_index"] for record in group]
        if indices != list(range(len(group))):
            raise ValueError(f"non-contiguous step indices in task {task_id}: {indices}")

        first = group[0]
        instruction = first.get("instruction")
        screenshot = first.get("screenshot")
        metadata = first.get("metadata")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"invalid instruction in task {task_id}")
        if screenshot is not None and not isinstance(screenshot, str):
            raise ValueError(f"invalid screenshot in task {task_id}")
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid metadata in task {task_id}")
        session_id = metadata.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(f"invalid session_id in task {task_id}")

        plan_steps: list[str] = []
        for record in group:
            if record.get("instruction") != instruction:
                raise ValueError(f"instruction mismatch in task {task_id}")
            if record.get("screenshot") != screenshot:
                raise ValueError(f"screenshot mismatch in task {task_id}")
            action = record.get("action")
            if not isinstance(action, dict):
                raise ValueError(f"invalid action object in task {task_id}")
            action_type = action.get("action_type")
            if not isinstance(action_type, str) or not action_type:
                raise ValueError(f"invalid action_type in task {task_id}")
            if action_type != ActionType.PLAN.value:
                continue
            element = action.get("element")
            if not isinstance(element, str) or not element.strip():
                raise ValueError(f"plan action has invalid element in task {task_id}")
            plan_steps.append(element.strip())

        if plan_steps:
            samples.append(
                {
                    "sample_id": task_id,
                    "source": SCREENAGENT_DATASET_NAME,
                    "session_id": session_id,
                    "instruction": instruction,
                    "screenshot": screenshot,
                    "plan_steps": plan_steps,
                    "num_steps": len(plan_steps),
                }
            )

    return samples


def build_screenagent_planner_dataset(
    split_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    split_dir = _require_directory(split_dir, field_name="split_dir")
    out = Path(output_dir) if output_dir is not None else split_dir.parent / "planner"

    split_manifest = _load_json_object(split_dir / "split_manifest.json")
    if split_manifest.get("session_overlap") != 0:
        raise RuntimeError("source split contains session leakage")
    record_counts = split_manifest.get("record_counts")
    if not isinstance(record_counts, dict):
        raise ValueError("split manifest is missing record counts")

    results: dict[str, list[dict[str, Any]]] = {}
    session_sets: dict[str, set[str]] = {}
    for split in ("train", "val"):
        records = _load_screenagent_records(split_dir / f"normal_{split}.jsonl")
        expected = record_counts.get(f"normal_{split}")
        if type(expected) is not int or expected < 0:
            raise ValueError(f"invalid split manifest count for normal_{split}")
        if len(records) != expected:
            raise RuntimeError(
                f"normal_{split} count mismatch: manifest={expected}, actual={len(records)}"
            )
        samples = build_planner_samples(records)
        if not samples:
            raise RuntimeError(f"no planner samples found for split {split}")
        results[split] = samples
        session_sets[split] = {sample["session_id"] for sample in samples}

    overlap = (
        session_sets["train"]
        & session_sets["val"]
    )

    if overlap:
        raise RuntimeError(
            "planner session leakage detected: "
            f"{sorted(overlap)}"
        )

    train_source_sessions = set(
        split_manifest.get(
            "train_session_ids",
            [],
        )
    )
    val_source_sessions = set(
        split_manifest.get(
            "val_session_ids",
            [],
        )
    )

    if not session_sets["train"].issubset(
        train_source_sessions
    ):
        raise RuntimeError(
            "planner train samples contain "
            "sessions outside the train split"
        )

    if not session_sets["val"].issubset(
        val_source_sessions
    ):
        raise RuntimeError(
            "planner validation samples contain "
            "sessions outside the validation split"
        )

    sample_counts = {split: len(samples) for split, samples in results.items()}
    plan_step_counts = {
        split: sum(sample["num_steps"] for sample in samples)
        for split, samples in results.items()
    }
    manifest = {
        "dataset": SCREENAGENT_DATASET_NAME,
        "schema_version": SCREENAGENT_SCHEMA_VERSION,
        "source_revision": split_manifest.get("source_revision"),
        "source_strategy": split_manifest.get("strategy"),
        "sample_counts": sample_counts,
        "plan_step_counts": plan_step_counts,
        "train_sessions": len(session_sets["train"]),
        "val_sessions": len(session_sets["val"]),
        "train_session_ids": sorted(
            session_sets["train"]
        ),
        "val_session_ids": sorted(
            session_sets["val"]
        ),
        "session_overlap": 0,
        "files": {
            "train": "planner_train.jsonl",
            "val": "planner_val.jsonl",
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    for split, samples in results.items():
        _atomic_write_jsonl(out / f"planner_{split}.jsonl", samples)
    _atomic_write_json(out / "planner_manifest.json", manifest)
    return manifest
