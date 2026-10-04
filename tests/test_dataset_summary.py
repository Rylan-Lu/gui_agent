from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui_agent.datasets.dataset_summary import (
    DatasetSummaryError,
    build_dataset_summary,
    write_dataset_summary,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _screenagent_manifest() -> dict:
    return {
        "dataset": "screenagent",
        "schema_version": 1,
        "source": {"revision": "screen-rev", "split": "train"},
        "raw": {"sessions": 203, "json_files": 3005, "images": 2004},
        "records": {"total": 5148, "normal": 3486, "negative": 1662},
        "action_distribution": {"plan": 2546, "click": 520},
        "quality": {"parse_failures": 0, "unknown_actions": 0},
    }


def _mind2web_manifest() -> dict:
    return {
        "dataset": "mind2web",
        "schema_version": 1,
        "source": {"revision": "mind-rev", "split": "train"},
        "raw": {
            "shards": 11,
            "tasks": 1009,
            "actions": 7775,
            "unique_task_ids": 1009,
            "unique_action_uids": 7775,
        },
        "action_distribution": {"click": 6513, "type_text": 936, "select": 326},
        "candidates": {
            "actions_with_positive_candidates": 7362,
            "actions_without_positive_candidates": 413,
        },
        "quality": {
            "parse_failures": 0,
            "duplicate_task_ids": 0,
            "duplicate_action_uids": 0,
            "exported_actions": 7775,
        },
    }


def _webarena_manifest() -> dict:
    return {
        "dataset": "webarena",
        "schema_version": 1,
        "source": {"revision": "web-rev", "split": "test"},
        "raw": {
            "tasks": 812,
            "unique_task_ids": 812,
            "task_id_min": 0,
            "task_id_max": 811,
        },
        "site_distribution": {"gitlab": 204, "shopping": 192},
        "evaluation": {
            "eval_type_distribution": {
                "program_html": 411,
                "string_match": 335,
                "url_match": 205,
            }
        },
        "quality": {
            "parse_failures": 0,
            "duplicate_task_ids": 0,
            "exported_tasks": 812,
        },
    }


def _build_processed_root(tmp_path: Path) -> Path:
    _write_json(tmp_path / "screenagent" / "manifest.json", _screenagent_manifest())
    _write_json(tmp_path / "mind2web" / "manifest.json", _mind2web_manifest())
    _write_json(tmp_path / "webarena" / "manifest.json", _webarena_manifest())
    return tmp_path


def test_build_dataset_summary(tmp_path: Path) -> None:
    root = _build_processed_root(tmp_path)
    summary = build_dataset_summary(root)

    assert summary["schema_version"] == 1
    assert set(summary["datasets"]) == {"screenagent", "mind2web", "webarena"}
    assert summary["datasets"]["screenagent"]["inventory"]["records"] == 5148
    assert summary["datasets"]["mind2web"]["inventory"]["actions"] == 7775
    assert summary["datasets"]["webarena"]["inventory"]["tasks"] == 812
    assert "No cross-dataset total" in summary["notes"]["aggregation_policy"]


def test_write_dataset_summary_is_json(tmp_path: Path) -> None:
    root = _build_processed_root(tmp_path)
    result = write_dataset_summary(root)
    output = root / "dataset_summary.json"

    assert output.is_file()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == result
    assert not (root / "dataset_summary.json.tmp").exists()


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    _write_json(tmp_path / "screenagent" / "manifest.json", _screenagent_manifest())
    _write_json(tmp_path / "mind2web" / "manifest.json", _mind2web_manifest())

    with pytest.raises(DatasetSummaryError, match="Missing dataset manifest"):
        build_dataset_summary(tmp_path)


def test_mind2web_count_mismatch_rejected(tmp_path: Path) -> None:
    root = _build_processed_root(tmp_path)
    manifest = _mind2web_manifest()
    manifest["quality"]["exported_actions"] = 7774
    _write_json(root / "mind2web" / "manifest.json", manifest)

    with pytest.raises(DatasetSummaryError, match="exported action count"):
        build_dataset_summary(root)


def test_screenagent_record_count_mismatch_rejected(tmp_path: Path) -> None:
    root = _build_processed_root(tmp_path)
    manifest = _screenagent_manifest()
    manifest["records"]["negative"] = 1661
    _write_json(root / "screenagent" / "manifest.json", manifest)

    with pytest.raises(DatasetSummaryError, match="record counts do not close"):
        build_dataset_summary(root)


def test_webarena_duplicate_id_summary_rejected(tmp_path: Path) -> None:
    root = _build_processed_root(tmp_path)
    manifest = _webarena_manifest()
    manifest["raw"]["unique_task_ids"] = 811
    _write_json(root / "webarena" / "manifest.json", manifest)

    with pytest.raises(DatasetSummaryError, match="task IDs are not unique"):
        build_dataset_summary(root)
