import json
from pathlib import Path

import pytest

from gui_agent.datasets.screenagent_pipeline import (
    audit_screenagent_source,
    build_planner_samples,
    build_screenagent_planner_dataset,
    export_screenagent_dataset,
    load_jsonl,
    split_screenagent_dataset,
    validate_source_audit,
)


def _write_source_file(
    root: Path,
    *,
    session_id: str,
    name: str,
    actions: list[dict],
    image_name: str = "screen.png",
) -> Path:
    session_dir = root / session_id
    image_dir = session_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / image_name).write_bytes(b"image")
    path = session_dir / name
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_prompt_en": f"Task for {session_id}",
                "video_width": 100,
                "video_height": 100,
                "saved_image_name": image_name,
                "actions": actions,
            }
        ),
        encoding="utf-8",
    )
    return path


def _plan(text: str) -> dict:
    return {"action_type": "PlanAction", "element": text}


def _click() -> dict:
    return {
        "action_type": "MouseAction",
        "mouse_action_type": "click",
        "mouse_button": "left",
        "mouse_position": {"width": 10, "height": 20},
    }


def _build_raw_dataset(root: Path) -> None:
    _write_source_file(root, session_id="session_a", name="a.json", actions=[_plan("Open app"), _click()], image_name="a.png")
    _write_source_file(root, session_id="session_a", name="a_neg_plan.json", actions=[_plan("Bad plan")], image_name="a_neg.png")
    _write_source_file(root, session_id="session_b", name="b.json", actions=[_plan("Search")], image_name="b.png")
    _write_source_file(root, session_id="session_c", name="c.json", actions=[_plan("Close app")], image_name="c.png")
    _write_source_file(root, session_id="session_c", name="empty.json", actions=[], image_name="empty.png")


def test_audit_source_counts(tmp_path):
    raw = tmp_path / "raw"
    _build_raw_dataset(raw)
    audit = audit_screenagent_source(raw)
    validate_source_audit(audit)
    assert audit["raw"]["sessions"] == 3
    assert audit["raw"]["json_files"] == 5
    assert audit["raw"]["images"] == 5
    assert audit["raw"]["empty_action_files"] == 1
    assert audit["records"] == {"total": 5, "normal": 4, "negative": 1}
    assert audit["action_distribution"]["plan"] == 4
    assert audit["action_distribution"]["click"] == 1


def test_export_creates_portable_records_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _build_raw_dataset(raw)
    manifest = export_screenagent_dataset(raw, out, source_revision="abc123")
    assert manifest["source"]["revision"] == "abc123"
    assert manifest["records"]["total"] == 5
    normal = load_jsonl(out / "normal.jsonl")
    negative = load_jsonl(out / "negative.jsonl")
    assert len(normal) == 4
    assert len(negative) == 1
    assert not Path(normal[0]["screenshot"]).is_absolute()
    assert not Path(normal[0]["metadata"]["source_file"]).is_absolute()
    persisted = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest


def test_split_is_session_level_and_deterministic(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _build_raw_dataset(raw)
    export_screenagent_dataset(raw, out, source_revision="abc123")
    first = split_screenagent_dataset(out, seed=42, validation_ratio=1 / 3)
    first_train = (out / "splits" / "normal_train.jsonl").read_bytes()
    first_val = (out / "splits" / "normal_val.jsonl").read_bytes()
    second = split_screenagent_dataset(out, seed=42, validation_ratio=1 / 3)
    assert first == second
    assert (out / "splits" / "normal_train.jsonl").read_bytes() == first_train
    assert (out / "splits" / "normal_val.jsonl").read_bytes() == first_val
    assert first["session_overlap"] == 0
    assert set(first["train_session_ids"]).isdisjoint(first["val_session_ids"])
    assert set(first["train_session_ids"]) | set(first["val_session_ids"]) == {
        "session_a",
        "session_b",
        "session_c",
    }


def test_split_uses_manifest_counts_not_hard_coded_snapshot(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _build_raw_dataset(raw)
    export_screenagent_dataset(raw, out, source_revision="abc123")
    manifest = split_screenagent_dataset(out, validation_ratio=1 / 3)
    assert sum(manifest["record_counts"].values()) == 5


def test_build_planner_dataset(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    _build_raw_dataset(raw)
    export_screenagent_dataset(raw, out, source_revision="abc123")
    split_screenagent_dataset(out, seed=42, validation_ratio=1 / 3)
    manifest = build_screenagent_planner_dataset(out / "splits")
    assert manifest["session_overlap"] == 0
    assert manifest["sample_counts"]["train"] > 0
    assert manifest["sample_counts"]["val"] > 0
    train = load_jsonl(out / "planner" / "planner_train.jsonl")
    val = load_jsonl(out / "planner" / "planner_val.jsonl")
    assert len(train) == manifest["sample_counts"]["train"]
    assert len(val) == manifest["sample_counts"]["val"]


def test_build_planner_samples_rejects_corrupt_action():
    records = [
        {
            "task_id": "task",
            "instruction": "Do thing",
            "step_index": 0,
            "screenshot": "x.png",
            "metadata": {"session_id": "s1"},
            "action": None,
        }
    ]
    with pytest.raises(ValueError, match="action"):
        build_planner_samples(records)


def test_build_planner_samples_rejects_non_contiguous_steps():
    records = [
        {
            "task_id": "task",
            "instruction": "Do thing",
            "step_index": 1,
            "screenshot": "x.png",
            "metadata": {"session_id": "s1"},
            "action": {"action_type": "plan", "element": "Do it"},
        }
    ]
    with pytest.raises(ValueError, match="non-contiguous"):
        build_planner_samples(records)


def test_source_integrity_detects_missing_image(tmp_path):
    raw = tmp_path / "raw"
    path = _write_source_file(raw, session_id="session_a", name="a.json", actions=[_plan("Open app")])
    data = json.loads(path.read_text(encoding="utf-8"))
    (path.parent / "images" / data["saved_image_name"]).unlink()
    audit = audit_screenagent_source(raw)
    with pytest.raises(RuntimeError, match="integrity"):
        validate_source_audit(audit)
