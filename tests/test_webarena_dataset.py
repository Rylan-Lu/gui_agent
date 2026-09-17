import json

import pytest

from gui_agent.datasets.webarena import (
    WebArenaTask,
    load_webarena_file,
    parse_webarena_task,
)


def make_task():
    return {
        "task_id": 2,
        "intent": "Check out the classification section",
        "sites": ["misc"],
        "start_url": "https://example.com",
        "require_login": False,
        "require_reset": False,
        "eval": {
            "eval_types": [],
        },
        "reference_action_sequence": {
            "action_set_tag": "playwright",
            "action_sequence": [
                "action_1",
                "action_2",
            ],
        },
    }


def test_parse_task():
    task = parse_webarena_task(make_task())

    assert isinstance(task, WebArenaTask)
    assert task.task_id == 2
    assert task.sites == ("misc",)
    assert task.reference_action_count == 2


def test_missing_reference_actions():
    raw = make_task()
    raw.pop("reference_action_sequence")

    task = parse_webarena_task(raw)

    assert task.reference_action_sequence is None
    assert task.reference_action_count == 0


def test_invalid_task_id():
    raw = make_task()
    raw["task_id"] = "2"

    with pytest.raises(ValueError):
        parse_webarena_task(raw)


def test_missing_intent():
    raw = make_task()
    raw["intent"] = ""

    with pytest.raises(ValueError):
        parse_webarena_task(raw)


def test_invalid_evaluation():
    raw = make_task()
    raw["eval"] = []

    with pytest.raises(ValueError):
        parse_webarena_task(raw)


def test_load_file(tmp_path):
    path = tmp_path / "2.json"

    path.write_text(
        json.dumps(make_task()),
        encoding="utf-8",
    )

    task = load_webarena_file(path)

    assert task.task_id == 2
    assert task.reference_action_count == 2
    assert task.source_file == str(path)