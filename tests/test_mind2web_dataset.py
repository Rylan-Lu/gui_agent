import json

import pytest

from gui_agent.datasets.mind2web import (
    parse_mind2web_action,
    parse_mind2web_task,
    load_mind2web_file,
)

from gui_agent.datasets.schema import ActionType


def test_click_action():
    action = parse_mind2web_action({
        "operation": {
            "op": "CLICK",
            "value": "",
        },
        "action_uid": "action_001",
    })

    assert action.action_type == ActionType.CLICK
    assert action.position is None


def test_type_action():
    action = parse_mind2web_action({
        "operation": {
            "op": "TYPE",
            "value": "GUI Agent",
        }
    })

    assert action.action_type == ActionType.TYPE_TEXT
    assert action.text == "GUI Agent"


def test_select_action():
    action = parse_mind2web_action({
        "operation": {
            "op": "SELECT",
            "value": "Option A",
        }
    })

    assert action.action_type == ActionType.SELECT
    assert action.raw_action == "SELECT"
    assert action.text == "Option A"
    assert action.metadata["operation"]["op"] == "SELECT"


def test_unknown_operation():
    with pytest.raises(ValueError):
        parse_mind2web_action({
            "operation": {
                "op": "UNKNOWN",
            }
        })


def test_task_conversion():
    task = {
        "annotation_id": "task_001",
        "confirmed_task": "Search for GUI Agent",
        "website": "example",
        "domain": "example.com",
        "actions": [
            {
                "action_uid": "a1",
                "operation": {
                    "op": "CLICK",
                    "value": "",
                },
            },
            {
                "action_uid": "a2",
                "operation": {
                    "op": "TYPE",
                    "value": "GUI Agent",
                },
            },
        ],
    }

    examples = parse_mind2web_task(task)

    assert len(examples) == 2
    assert examples[0].source == "mind2web"
    assert examples[0].screenshot is None
    assert examples[1].step_index == 1
    assert len(examples[1].history) == 1


def test_load_file(tmp_path):
    path = tmp_path / "train.json"

    task = {
        "annotation_id": "task_001",
        "confirmed_task": "Open browser",
        "actions": [
            {
                "operation": {
                    "op": "CLICK",
                    "value": "",
                }
            }
        ],
    }

    path.write_text(
        json.dumps([task]),
        encoding="utf-8",
    )

    examples = load_mind2web_file(path)

    assert len(examples) == 1
    assert examples[0].task_id == "task_001"