import json

import pytest

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

    assert (
        action.action_type
        == ActionType.PLAN
    )

    assert (
        action.element
        == "Open browser"
    )


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

    assert (
        action.action_type
        == ActionType.CLICK
    )

    assert (
        action.position
        == Point(
            487,
            193,
        )
    )

    assert (
        action.mouse_button
        == "left"
    )


@pytest.mark.parametrize(
    "mouse_type,repeat,expected",
    [
        (
            "scroll_down",
            10,
            -10,
        ),
        (
            "scroll_up",
            3,
            3,
        ),
    ],
)
def test_parse_scroll_repeat(
    mouse_type,
    repeat,
    expected,
):
    action = parse_screenagent_action(
        {
            "action_type": (
                "MouseAction"
            ),
            "mouse_action_type": (
                mouse_type
            ),
            "mouse_position": None,
            "scroll_repeat": repeat,
        }
    )

    assert (
        action.action_type
        == ActionType.SCROLL
    )

    assert (
        action.scroll_delta
        == expected
    )

    assert (
        action.position
        is None
    )

    assert (
        action.metadata[
            "scroll_repeat"
        ]
        == repeat
    )


@pytest.mark.parametrize(
    "repeat",
    [
        None,
        0,
        -1,
        True,
        1.5,
        "10",
    ],
)
def test_invalid_scroll_repeat_rejected(
    repeat,
):
    with pytest.raises(
        ValueError,
        match="scroll_repeat",
    ):
        parse_screenagent_action(
            {
                "action_type": (
                    "MouseAction"
                ),
                "mouse_action_type": (
                    "scroll_down"
                ),
                "mouse_position": (
                    None
                ),
                "scroll_repeat": (
                    repeat
                ),
            }
        )


def test_drag_preserves_only_available_endpoint():
    action = parse_screenagent_action(
        {
            "action_type": (
                "MouseAction"
            ),
            "mouse_action_type": (
                "drag"
            ),
            "mouse_button": "left",
            "mouse_position": {
                "width": 573,
                "height": 407,
            },
        }
    )

    assert (
        action.action_type
        == ActionType.DRAG
    )

    assert (
        action.position
        is None
    )

    assert (
        action.end_position
        == Point(
            573,
            407,
        )
    )

    assert (
        action.metadata[
            "drag_start_available"
        ]
        is False
    )


@pytest.mark.parametrize(
    "point",
    [
        None,
        [],
        {
            "width": 1,
        },
        {
            "height": 2,
        },
        {
            "width": True,
            "height": 2,
        },
        {
            "width": "1",
            "height": 2,
        },
        {
            "width": float(
                "inf"
            ),
            "height": 2,
        },
    ],
)
def test_required_mouse_position_must_be_valid(
    point,
):
    with pytest.raises(
        ValueError,
        match="mouse_position",
    ):
        parse_screenagent_action(
            {
                "action_type": (
                    "MouseAction"
                ),
                "mouse_action_type": (
                    "click"
                ),
                "mouse_position": (
                    point
                ),
            }
        )


def test_parse_keyboard_text():
    action = parse_screenagent_action(
        {
            "action_type": (
                "KeyboardAction"
            ),
            "keyboard_action_type": (
                "text"
            ),
            "keyboard_text": (
                "Von Neumann"
            ),
        }
    )

    assert (
        action.action_type
        == ActionType.TYPE_TEXT
    )

    assert (
        action.text
        == "Von Neumann"
    )


def test_parse_keyboard_single_press():
    action = parse_screenagent_action(
        {
            "action_type": (
                "KeyboardAction"
            ),
            "keyboard_action_type": (
                "press"
            ),
            "keyboard_key": (
                "Return"
            ),
        }
    )

    assert (
        action.action_type
        == ActionType.KEY_PRESS
    )

    assert (
        action.keys
        == ("Return",)
    )


def test_numeric_looking_key_remains_string():
    action = parse_screenagent_action(
        {
            "action_type": (
                "KeyboardAction"
            ),
            "keyboard_action_type": (
                "press"
            ),
            "keyboard_key": "2",
        }
    )

    assert (
        action.action_type
        == ActionType.KEY_PRESS
    )

    assert (
        action.keys
        == ("2",)
    )


def test_parse_keyboard_list_as_hotkey():
    action = parse_screenagent_action(
        {
            "action_type": (
                "KeyboardAction"
            ),
            "keyboard_action_type": (
                "press"
            ),
            "keyboard_key": [
                "Control_L",
                "s",
            ],
        }
    )

    assert (
        action.action_type
        == ActionType.HOTKEY
    )

    assert (
        action.keys
        == (
            "Control_L",
            "s",
        )
    )


def test_parse_plus_encoded_hotkey():
    action = parse_screenagent_action(
        {
            "action_type": (
                "KeyboardAction"
            ),
            "keyboard_action_type": (
                "press"
            ),
            "keyboard_key": (
                "Ctrl+S"
            ),
        }
    )

    assert (
        action.action_type
        == ActionType.HOTKEY
    )

    assert (
        action.keys
        == (
            "Ctrl",
            "S",
        )
    )


@pytest.mark.parametrize(
    "key",
    [
        None,
        "",
        "   ",
        [],
        [
            "Control_L",
            "",
        ],
        [
            "Control_L",
            1,
        ],
        2,
    ],
)
def test_invalid_keyboard_key_rejected(
    key,
):
    with pytest.raises(
        ValueError,
        match="keyboard_key",
    ):
        parse_screenagent_action(
            {
                "action_type": (
                    "KeyboardAction"
                ),
                "keyboard_action_type": (
                    "press"
                ),
                "keyboard_key": key,
            }
        )


def test_parse_wait():
    action = parse_screenagent_action(
        {
            "action_type": (
                "WaitAction"
            ),
            "wait_time": 0.5,
        }
    )

    assert (
        action.action_type
        == ActionType.WAIT
    )

    assert (
        action.wait_seconds
        == 0.5
    )


@pytest.mark.parametrize(
    "wait_time",
    [
        None,
        True,
        -1,
        "0.5",
        float("inf"),
    ],
)
def test_invalid_wait_rejected(
    wait_time,
):
    with pytest.raises(
        ValueError,
        match="wait_time",
    ):
        parse_screenagent_action(
            {
                "action_type": (
                    "WaitAction"
                ),
                "wait_time": (
                    wait_time
                ),
            }
        )


def test_parse_evaluation():
    action = parse_screenagent_action(
        {
            "action_type": (
                "EvaluateSubTaskAction"
            ),
            "situation": (
                "need_retry"
            ),
            "advice": (
                "Try again"
            ),
        }
    )

    assert (
        action.action_type
        == ActionType.EVALUATE
    )

    assert (
        action.evaluation_status
        == EvaluationStatus.NEED_RETRY
    )

    assert (
        action.advice
        == "Try again"
    )


def test_unknown_action_is_preserved_as_other():
    raw = {
        "action_type": (
            "FutureAction"
        ),
        "payload": 123,
    }

    action = (
        parse_screenagent_action(
            raw
        )
    )

    assert (
        action.action_type
        == ActionType.OTHER
    )

    assert (
        action.raw_action
        == "FutureAction"
    )

    assert (
        action.metadata
        == raw
    )


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {
            "action_type": None,
        },
        {
            "action_type": "",
        },
    ],
)
def test_missing_action_type_rejected(
    raw,
):
    with pytest.raises(
        ValueError,
        match="action_type",
    ):
        parse_screenagent_action(
            raw
        )


def _write_screenagent_file(
    tmp_path,
    data,
):
    session_dir = (
        tmp_path
        / "session_001"
    )

    session_dir.mkdir(
        exist_ok=True
    )

    images_dir = (
        session_dir
        / "images"
    )

    images_dir.mkdir(
        exist_ok=True
    )

    json_path = (
        session_dir
        / "sample_translate.json"
    )

    json_path.write_text(
        json.dumps(
            data
        ),
        encoding="utf-8",
    )

    return json_path


def _base_file_data(
    actions,
):
    return {
        "session_id": (
            "session_001"
        ),
        "task_prompt_en": (
            "Search for GUI Agent"
        ),
        "video_width": 1024,
        "video_height": 768,
        "saved_image_name": (
            "screen.jpg"
        ),
        "actions": actions,
    }


def test_load_screenagent_file(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                [
                    {
                        "action_type": (
                            "PlanAction"
                        ),
                        "element": (
                            "Open browser"
                        ),
                    },
                    {
                        "action_type": (
                            "KeyboardAction"
                        ),
                        "keyboard_action_type": (
                            "text"
                        ),
                        "keyboard_text": (
                            "GUI Agent"
                        ),
                    },
                ]
            ),
        )
    )

    examples = (
        load_screenagent_file(
            json_path
        )
    )

    assert len(
        examples
    ) == 2

    assert (
        examples[0].source
        == "screenagent"
    )

    assert (
        examples[0].step_index
        == 0
    )

    assert (
        examples[0].history
        == ()
    )

    assert (
        examples[1].step_index
        == 1
    )

    assert (
        len(
            examples[1].history
        )
        == 1
    )

    assert (
        examples[1]
        .action
        .action_type
        == ActionType.TYPE_TEXT
    )


def test_empty_actions_is_valid(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                []
            ),
        )
    )

    assert (
        load_screenagent_file(
            json_path
        )
        == []
    )


def test_missing_actions_rejected(
    tmp_path,
):
    data = (
        _base_file_data(
            []
        )
    )

    del data[
        "actions"
    ]

    json_path = (
        _write_screenagent_file(
            tmp_path,
            data,
        )
    )

    with pytest.raises(
        ValueError,
        match="missing actions",
    ):
        load_screenagent_file(
            json_path
        )


def test_null_actions_rejected(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                None
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="must not be null",
    ):
        load_screenagent_file(
            json_path
        )


def test_non_list_actions_rejected(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                {}
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="must be a list",
    ):
        load_screenagent_file(
            json_path
        )


def test_invalid_action_item_is_not_silently_dropped(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                [
                    {
                        "action_type": (
                            "PlanAction"
                        ),
                        "element": (
                            "Open browser"
                        ),
                    },
                    "bad-action",
                ]
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="action_index=1",
    ):
        load_screenagent_file(
            json_path
        )


def test_invalid_action_reports_file_and_index(
    tmp_path,
):
    json_path = (
        _write_screenagent_file(
            tmp_path,
            _base_file_data(
                [
                    {
                        "action_type": (
                            "KeyboardAction"
                        ),
                        "keyboard_action_type": (
                            "press"
                        ),
                        "keyboard_key": (
                            None
                        ),
                    }
                ]
            ),
        )
    )

    with pytest.raises(
        ValueError
    ) as exc_info:
        load_screenagent_file(
            json_path
        )

    message = str(
        exc_info.value
    )

    assert (
        str(json_path)
        in message
    )

    assert (
        "action_index=0"
        in message
    )

    assert (
        "keyboard_key"
        in message
    )