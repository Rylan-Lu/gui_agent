import json

import pytest

from gui_agent.datasets.webarena_pipeline import export_webarena_dataset


def make_task(task_id, site="shopping_admin", eval_type="string_match"):
    evaluation = {
        "eval_types": [eval_type],
        "reference_answers": {"exact_match": f"answer-{task_id}"},
        "reference_url": "",
        "program_html": [],
    }

    if eval_type == "program_html":
        evaluation = {
            "eval_types": ["program_html"],
            "reference_answers": None,
            "reference_url": None,
            "program_html": [
                {
                    "url": "last",
                    "locator": "",
                    "required_contents": {
                        "must_include": ["target"],
                    },
                }
            ],
        }

    return {
        "sites": [site],
        "task_id": task_id,
        "require_login": True,
        "storage_state": None,
        "start_url": f"__{site.upper()}__",
        "geolocation": None,
        "intent_template": "Do {{thing}}",
        "instantiation_dict": {"thing": str(task_id)},
        "intent": f"Do {task_id}",
        "require_reset": False,
        "eval": evaluation,
        "intent_template_id": task_id,
    }


def test_export_collection(tmp_path):
    raw_path = tmp_path / "test.raw.json"
    output_dir = tmp_path / "processed"

    raw_path.write_text(
        json.dumps(
            [
                make_task(0, "shopping_admin", "string_match"),
                make_task(1, "gitlab", "url_match"),
                make_task(2, "shopping", "program_html"),
            ]
        ),
        encoding="utf-8",
    )

    manifest = export_webarena_dataset(
        raw_path,
        output_dir,
        source_revision="abc123",
    )

    assert manifest["raw"] == {
        "tasks": 3,
        "unique_task_ids": 3,
        "task_id_min": 0,
        "task_id_max": 2,
    }
    assert manifest["quality"]["exported_tasks"] == 3
    assert manifest["quality"]["duplicate_task_ids"] == 0
    assert manifest["reference_actions"] == {
        "tasks_with_reference_action_sequence": 0,
        "total_reference_actions": 0,
    }

    lines = (output_dir / "tasks.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    assert [record["task_id"] for record in records] == [0, 1, 2]
    assert records[2]["evaluation"]["program_html"][0]["url"] == "last"

    disk_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert disk_manifest == manifest


def test_duplicate_task_id_fails_before_manifest(tmp_path):
    raw_path = tmp_path / "test.raw.json"
    output_dir = tmp_path / "processed"

    raw_path.write_text(
        json.dumps([make_task(1), make_task(1)]),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Duplicate WebArena task_id"):
        export_webarena_dataset(
            raw_path,
            output_dir,
            source_revision="abc123",
        )

    assert not (output_dir / "manifest.json").exists()


def test_source_revision_required(tmp_path):
    raw_path = tmp_path / "test.raw.json"
    raw_path.write_text(json.dumps([make_task(0)]), encoding="utf-8")

    with pytest.raises(ValueError, match="source_revision"):
        export_webarena_dataset(
            raw_path,
            tmp_path / "processed",
            source_revision="",
        )
