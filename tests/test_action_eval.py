import pytest

from gui_agent.evaluation.action_eval import (
    build_action_policy_prompt,
    parse_json_object,
    score_prediction,
    select_samples,
    validate_action_prediction,
)


def test_parse_json_object():
    result = parse_json_object(
        '{"action_type":"wait",'
        '"wait_seconds":0.5}'
    )

    assert result["action_type"] == "wait"


def test_parse_json_object_accepts_fence():
    result = parse_json_object(
        '```json\n'
        '{"action_type":"scroll",'
        '"scroll_delta":-1}\n'
        '```'
    )

    assert result == {
        "action_type": "scroll",
        "scroll_delta": -1,
    }


def test_click_prediction_is_valid():
    prediction = (
        validate_action_prediction(
            {
                "action_type": "click",
                "position": {
                    "x": 100,
                    "y": 200,
                },
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )
    )

    assert prediction["position"] == {
        "x": 100,
        "y": 200,
    }


def test_click_outside_screen_rejected():
    with pytest.raises(ValueError):
        validate_action_prediction(
            {
                "action_type": "click",
                "position": {
                    "x": 2000,
                    "y": 100,
                },
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )


def test_key_press_is_normalized():
    prediction = (
        validate_action_prediction(
            {
                "action_type": "key_press",
                "keys": ["Return"],
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )
    )

    assert prediction == {
        "action_type": "key_press",
        "keys": ["enter"],
    }


def test_coordinate_score():
    score = score_prediction(
        {
            "action_type": "click",
            "position": {
                "x": 10,
                "y": 10,
            },
        },
        {
            "action_type": "click",
            "position": {
                "x": 13,
                "y": 14,
            },
        },
    )

    assert score[
        "action_type_correct"
    ]

    assert score[
        "coordinate_error"
    ] == pytest.approx(5.0)

    assert score[
        "mouse_button_correct"
    ] is True

    assert score[
        "success_at_20px"
    ]


def test_type_text_exact_score():
    score = score_prediction(
        {
            "action_type": "type_text",
            "text": "hello",
        },
        {
            "action_type": "type_text",
            "text": "hello",
        },
    )

    assert score[
        "exact_action_correct"
    ]


def test_build_action_policy_prompt():
    prompt = build_action_policy_prompt(
        {
            "instruction": "Search for cats",
            "history": [],
            "screen_size": {
                "width": 1024,
                "height": 768,
            },
        }
    )

    assert "Search for cats" in prompt
    assert "1024x768" in prompt


def test_select_samples_limit():
    samples = [
        {
            "sample_id": str(index),
            "target": {
                "action_type": (
                    "click"
                    if index % 2 == 0
                    else "type_text"
                )
            },
        }
        for index in range(10)
    ]

    selected = select_samples(
        samples,
        limit=5,
        seed=42,
    )

    assert len(selected) == 5


def test_right_click_prediction_is_valid():
    prediction = (
        validate_action_prediction(
            {
                "action_type": "click",
                "position": {
                    "x": 100,
                    "y": 200,
                },
                "mouse_button": "right",
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )
    )

    assert prediction == {
        "action_type": "click",
        "position": {
            "x": 100,
            "y": 200,
        },
        "mouse_button": "right",
    }


def test_left_click_is_canonicalized_without_button():
    prediction = (
        validate_action_prediction(
            {
                "action_type": "click",
                "position": {
                    "x": 100,
                    "y": 200,
                },
                "mouse_button": "left",
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )
    )

    assert prediction == {
        "action_type": "click",
        "position": {
            "x": 100,
            "y": 200,
        },
    }


def test_invalid_mouse_button_rejected():
    with pytest.raises(ValueError):
        validate_action_prediction(
            {
                "action_type": "click",
                "position": {
                    "x": 100,
                    "y": 200,
                },
                "mouse_button": "side",
            },
            screen_size={
                "width": 1024,
                "height": 768,
            },
        )


def test_wrong_mouse_button_is_not_exact():
    score = score_prediction(
        {
            "action_type": "click",
            "position": {
                "x": 100,
                "y": 200,
            },
            "mouse_button": "right",
        },
        {
            "action_type": "click",
            "position": {
                "x": 100,
                "y": 200,
            },
        },
    )

    assert score[
        "action_type_correct"
    ]

    assert score[
        "mouse_button_correct"
    ] is False

    assert not score[
        "exact_action_correct"
    ]

    assert not score[
        "success_at_20px"
    ]

    assert not score[
        "success_at_50px"
    ]


def test_right_click_point_threshold_success():
    score = score_prediction(
        {
            "action_type": "click",
            "position": {
                "x": 100,
                "y": 200,
            },
            "mouse_button": "right",
        },
        {
            "action_type": "click",
            "position": {
                "x": 110,
                "y": 205,
            },
            "mouse_button": "right",
        },
    )

    assert score[
        "action_type_correct"
    ]

    assert score[
        "mouse_button_correct"
    ] is True

    assert score[
        "success_at_20px"
    ]

    assert score[
        "success_at_50px"
    ]

    assert not score[
        "exact_action_correct"
    ]