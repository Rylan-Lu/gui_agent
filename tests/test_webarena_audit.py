import json

from gui_agent.datasets.webarena_audit import (
    audit_webarena_files,
    discover_webarena_files,
)


def make_task(task_id, site="shopping_admin", eval_types=None):
    if eval_types is None:
        eval_types = ["url_match"]

    return {
        "sites": [site],
        "task_id": task_id,
        "require_login": True,
        "storage_state": None,
        "start_url": f"__{site.upper()}__",
        "geolocation": None,
        "intent_template": "Complete {{thing}}",
        "instantiation_dict": {"thing": "test"},
        "intent": "Complete a test task",
        "require_reset": False,
        "eval": {
            "eval_types": eval_types,
            "reference_answers": None,
            "reference_url": "",
            "program_html": [],
        },
        "intent_template_id": task_id,
    }


def save_task(path, task):
    path.write_text(json.dumps(task), encoding="utf-8")


def test_multiple_files(tmp_path):
    save_task(tmp_path / "1.json", make_task(1))
    save_task(tmp_path / "2.json", make_task(2))

    report = audit_webarena_files(discover_webarena_files(tmp_path))

    assert report["total_files"] == 2
    assert report["raw_tasks"] == 2
    assert report["parsed_tasks"] == 2
    assert report["failed_tasks"] == 0
    assert report["reference_actions"] == 0
    assert report["status"] == "PASS"


def test_collection_file(tmp_path):
    path = tmp_path / "test.raw.json"
    path.write_text(
        json.dumps([make_task(0), make_task(1)]),
        encoding="utf-8",
    )

    report = audit_webarena_files(discover_webarena_files(path))

    assert report["total_files"] == 1
    assert report["raw_tasks"] == 2
    assert report["parsed_tasks"] == 2
    assert report["failed_tasks"] == 0
    assert report["status"] == "PASS"


def test_duplicate_task_id(tmp_path):
    save_task(tmp_path / "1.json", make_task(1))
    save_task(tmp_path / "2.json", make_task(1))

    report = audit_webarena_files(discover_webarena_files(tmp_path))

    assert report["parsed_tasks"] == 1
    assert report["failed_tasks"] == 1
    assert report["status"] == "FAILED"
    assert "Duplicate task_id" in report["errors"][0]["error"]


def test_invalid_task_does_not_affect_statistics(tmp_path):
    save_task(tmp_path / "1.json", make_task(1, site="gitlab"))

    invalid = make_task(2, site="shopping")
    invalid["eval"]["eval_types"] = "invalid"
    save_task(tmp_path / "2.json", invalid)

    report = audit_webarena_files(discover_webarena_files(tmp_path))

    assert report["parsed_tasks"] == 1
    assert report["failed_tasks"] == 1
    assert report["site_distribution"] == {"gitlab": 1}
    assert report["status"] == "FAILED"
