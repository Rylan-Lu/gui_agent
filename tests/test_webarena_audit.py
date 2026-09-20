import json

from scripts.audit_webarena import audit, discover_files


def make_task(task_id, site="misc", eval_types=None):
    if eval_types is None:
        eval_types = ["url_match"]

    return {
        "task_id": task_id,
        "intent": "Complete a test task",
        "sites": [site],
        "start_url": "https://example.com",
        "require_login": False,
        "require_reset": False,
        "eval": {"eval_types": eval_types},
        "reference_action_sequence": {
            "action_set_tag": "playwright",
            "action_sequence": ["click", "stop"],
        },
    }


def save_task(path, task):
    path.write_text(
        json.dumps(task),
        encoding="utf-8",
    )


def test_multiple_files(tmp_path):
    save_task(tmp_path / "1.json", make_task(1))
    save_task(tmp_path / "2.json", make_task(2))

    report = audit(discover_files(tmp_path))

    assert report["total_files"] == 2
    assert report["parsed_tasks"] == 2
    assert report["failed_files"] == 0
    assert report["reference_actions"] == 4
    assert report["status"] == "PASS"


def test_duplicate_task_id(tmp_path):
    save_task(tmp_path / "1.json", make_task(1))
    save_task(tmp_path / "2.json", make_task(1))

    report = audit(discover_files(tmp_path))

    assert report["parsed_tasks"] == 1
    assert report["failed_files"] == 1
    assert report["status"] == "FAILED"
    assert "Duplicate task_id" in report["errors"][0]["error"]


def test_invalid_task_does_not_affect_statistics(tmp_path):
    save_task(
        tmp_path / "1.json",
        make_task(1, site="misc"),
    )

    save_task(
        tmp_path / "2.json",
        make_task(
            2,
            site="invalid_site",
            eval_types="invalid",
        ),
    )

    report = audit(discover_files(tmp_path))

    assert report["parsed_tasks"] == 1
    assert report["failed_files"] == 1
    assert report["site_distribution"] == {"misc": 1}
    assert report["reference_actions"] == 2