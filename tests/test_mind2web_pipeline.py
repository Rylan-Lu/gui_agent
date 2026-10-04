import json

import pytest

from gui_agent.datasets.mind2web_pipeline import (
    discover_mind2web_shards,
    export_mind2web_dataset,
)


def candidate(node_id="1", *, original=True):
    return {
        "tag": "button",
        "attributes": json.dumps(
            {
                "backend_node_id": node_id,
                "bounding_box_rect": "1,2,3,4",
                "aria_label": "Open",
            }
        ),
        "is_original_target": original,
        "is_top_level_target": True,
        "backend_node_id": node_id,
    }


def action(uid, *, op="CLICK", pos=True):
    return {
        "action_uid": uid,
        "raw_html": "<html>raw</html>",
        "cleaned_html": "<html>cleaned</html>",
        "operation": {
            "original_op": op,
            "value": "" if op == "CLICK" else "value",
            "op": op,
        },
        "pos_candidates": [candidate(uid)] if pos else [],
        "neg_candidates": [candidate(f"neg-{uid}", original=False)],
    }


def task(task_id, actions):
    return {
        "website": "example",
        "domain": "Shopping",
        "subdomain": "Retail",
        "annotation_id": task_id,
        "confirmed_task": f"Task {task_id}",
        "action_reprs": [f"repr-{i}" for i in range(len(actions))],
        "actions": actions,
    }


def write_shard(path, tasks):
    path.write_text(
        json.dumps(tasks),
        encoding="utf-8",
    )


def test_discover_shards_uses_numeric_order(tmp_path):
    write_shard(tmp_path / "train_10.json", [])
    write_shard(tmp_path / "train_2.json", [])
    write_shard(tmp_path / "train_0.json", [])

    shards = discover_mind2web_shards(tmp_path)
    assert [path.name for path in shards] == [
        "train_0.json",
        "train_2.json",
        "train_10.json",
    ]


def test_export_dataset_builds_actions_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()

    write_shard(
        raw / "train_0.json",
        [
            task(
                "task-0",
                [
                    action("a0", op="CLICK"),
                    action("a1", op="TYPE", pos=False),
                ],
            )
        ],
    )
    write_shard(
        raw / "train_1.json",
        [task("task-1", [action("a2", op="SELECT")])],
    )

    manifest = export_mind2web_dataset(
        raw,
        out,
        source_revision="abc123",
    )

    assert manifest["raw"] == {
        "shards": 2,
        "tasks": 2,
        "actions": 3,
        "unique_task_ids": 2,
        "unique_action_uids": 3,
    }
    assert manifest["action_distribution"] == {
        "click": 1,
        "select": 1,
        "type_text": 1,
    }
    assert manifest["candidates"]["actions_without_positive_candidates"] == 1
    assert manifest["source"]["revision"] == "abc123"

    lines = (out / "actions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert records[0]["task_id"] == "task-0"
    assert records[0]["metadata"]["source_file"] == "train_0.json"

    # Compact export: source HTML is validated but not duplicated.
    serialized = (out / "actions.jsonl").read_text(encoding="utf-8")
    assert "<html>raw</html>" not in serialized
    assert "<html>cleaned</html>" not in serialized

    assert (out / "manifest.json").is_file()
    assert not (out / "actions.jsonl.tmp").exists()


def test_duplicate_annotation_id_across_shards_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()

    write_shard(raw / "train_0.json", [task("same", [action("a0")])])
    write_shard(raw / "train_1.json", [task("same", [action("a1")])])

    with pytest.raises(ValueError, match="Duplicate annotation_id"):
        export_mind2web_dataset(
            raw,
            out,
            source_revision="abc123",
        )

    assert not (out / "actions.jsonl").exists()
    assert not (out / "actions.jsonl.tmp").exists()


def test_duplicate_action_uid_across_shards_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()

    write_shard(raw / "train_0.json", [task("t0", [action("same")])])
    write_shard(raw / "train_1.json", [task("t1", [action("same")])])

    with pytest.raises(ValueError, match="Duplicate action_uid"):
        export_mind2web_dataset(
            raw,
            out,
            source_revision="abc123",
        )

    assert not (out / "actions.jsonl").exists()
