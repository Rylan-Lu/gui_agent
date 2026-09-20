import json

from gui_agent.datasets.schema import (
    ActionType,
    EvaluationStatus,
    Point,
)

from gui_agent.datasets.screenagent import (
    load_screenagent_file,
    parse_screenagent_action,
)


def test_parse_plan_action():
    action = parse_screenagent_action(
        {
            "action_type": "PlanAction",
            "element": "Open browser",
        }
    )

    assert action.action_type == ActionType.PLAN
    assert action.element == "Open browser"


def test_parse_mouse_click():
    action = parse_screenagent_action(
        {
            "action_type": "MouseAction",
            "mouse_action_type": "click",
            "mouse_button": "left",
            "mouse_position": {
                "width": 487,
                "height": 193,
            },
        }
    )

    assert action.action_type == ActionType.CLICK
    assert action.position == Point(487, 193)
    assert action.mouse_button == "left"


def test_parse_keyboard_text():
    action = parse_screenagent_action(
        {
            "action_type": "KeyboardAction",
            "keyboard_action_type": "text",
            "keyboard_text": "Von Neumann",
        }
    )

    assert action.action_type == ActionType.TYPE_TEXT
    assert action.text == "Von Neumann"


def test_parse_keyboard_press():
    action = parse_screenagent_action(
        {
            "action_type": "KeyboardAction",
            "keyboard_action_type": "press",
            "keyboard_key": "enter",
        }
    )

    assert action.action_type == ActionType.KEY_PRESS
    assert action.keys == ("enter",)


def test_parse_wait():
    action = parse_screenagent_action(
        {
            "action_type": "WaitAction",
            "wait_time": 0.5,
        }
    )

    assert action.action_type == ActionType.WAIT
    assert action.wait_seconds == 0.5


def test_parse_evaluation():
    action = parse_screenagent_action(
        {
            "action_type": "EvaluateSubTaskAction",
            "situation": "need_retry",
            "advice": "Try again",
        }
    )

    assert action.action_type == ActionType.EVALUATE

    assert (
        action.evaluation_status
        == EvaluationStatus.NEED_RETRY
    )

    assert action.advice == "Try again"


def test_load_screenagent_file(tmp_path):
    session_dir = tmp_path / "session_001"
    session_dir.mkdir()

    images_dir = session_dir / "images"
    images_dir.mkdir()

    data = {
        "session_id": "session_001",
        "task_prompt_en": "Search for GUI Agent",
        "video_width": 1024,
        "video_height": 768,
        "saved_image_name": "screen.jpg",
        "actions": [
            {
                "action_type": "PlanAction",
                "element": "Open browser",
            },
            {
                "action_type": "KeyboardAction",
                "keyboard_action_type": "text",
                "keyboard_text": "GUI Agent",
            },
        ],
    }

    json_path = session_dir / "sample_translate.json"

    json_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    examples = load_screenagent_file(
        json_path
    )

    assert len(examples) == 2

    assert examples[0].source == "screenagent"
    assert examples[0].step_index == 0
    assert examples[0].history == ()

    assert examples[1].step_index == 1
    assert len(examples[1].history) == 1

    assert (
        examples[1].action.action_type
        == ActionType.TYPE_TEXT
    )