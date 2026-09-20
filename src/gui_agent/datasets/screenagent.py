from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gui_agent.datasets.schema import (
    ActionType,
    EvaluationStatus,
    GUIAction,
    GUIExample,
    Point,
)


def _parse_point(value: Any) -> Point | None:
    """
    Convert ScreenAgent's mouse_position into a normalized Point object.

    ScreenAgent uses:
        {
            "width": x,
            "height": y
        }
    """

    if not isinstance(value, dict):
        return None

    x = value.get("width")
    y = value.get("height")

    if x is None or y is None:
        return None

    return Point(
        x=float(x),
        y=float(y),
    )


def parse_screenagent_action(raw: dict[str, Any]) -> GUIAction:
    """Convert one raw ScreenAgent action into GUIAction."""

    raw_type = raw.get("action_type")

    if raw_type == "PlanAction":
        return GUIAction(
            action_type=ActionType.PLAN,
            element=raw.get("element"),
            raw_action=raw_type,
        )

    if raw_type == "MouseAction":
        mouse_type = raw.get("mouse_action_type")
        position = _parse_point(raw.get("mouse_position"))

        action_map = {
            "click": ActionType.CLICK,
            "double_click": ActionType.DOUBLE_CLICK,
            "move": ActionType.MOVE,
            "drag": ActionType.DRAG,
            "down": ActionType.MOUSE_DOWN,
            "up": ActionType.MOUSE_UP,
            "scroll_down": ActionType.SCROLL,
            "scroll_up": ActionType.SCROLL,
        }

        action_type = action_map.get(
            mouse_type,
            ActionType.OTHER,
        )

        scroll_delta = None

        if mouse_type == "scroll_down":
            scroll_delta = -1
        elif mouse_type == "scroll_up":
            scroll_delta = 1

        return GUIAction(
            action_type=action_type,
            position=position,
            mouse_button=raw.get("mouse_button") or None,
            scroll_delta=scroll_delta,
            raw_action=raw_type,
            metadata={
                "mouse_action_type": mouse_type,
                "scroll_repeat": raw.get("scroll_repeat"),
            },
        )

    if raw_type == "KeyboardAction":
        keyboard_type = raw.get("keyboard_action_type")

        if keyboard_type == "text":
            return GUIAction(
                action_type=ActionType.TYPE_TEXT,
                text=raw.get("keyboard_text"),
                raw_action=raw_type,
                metadata={
                    "keyboard_action_type": keyboard_type,
                },
            )

        if keyboard_type == "press":
            key = raw.get("keyboard_key")

            return GUIAction(
                action_type=ActionType.KEY_PRESS,
                keys=(str(key),) if key is not None else (),
                raw_action=raw_type,
                metadata={
                    "keyboard_action_type": keyboard_type,
                },
            )

        return GUIAction(
            action_type=ActionType.OTHER,
            raw_action=raw_type,
            metadata=dict(raw),
        )

    if raw_type == "WaitAction":
        wait_time = raw.get("wait_time")

        return GUIAction(
            action_type=ActionType.WAIT,
            wait_seconds=(
                float(wait_time)
                if wait_time is not None
                else None
            ),
            raw_action=raw_type,
        )

    if raw_type == "EvaluateSubTaskAction":
        raw_situation = raw.get("situation")

        try:
            status = EvaluationStatus(raw_situation)
        except (ValueError, TypeError):
            status = None

        return GUIAction(
            action_type=ActionType.EVALUATE,
            evaluation_status=status,
            advice=raw.get("advice"),
            raw_action=raw_type,
            metadata={
                "raw_situation": raw_situation,
            },
        )

    return GUIAction(
        action_type=ActionType.OTHER,
        raw_action=str(raw_type) if raw_type is not None else None,
        metadata=dict(raw),
    )


def load_screenagent_file(path: str | Path) -> list[GUIExample]:
    """
    Load one ScreenAgent JSON file.

    One ScreenAgent JSON may contain multiple actions, so this function
    returns one GUIExample per action.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    session_id = str(
        data.get("session_id")
        or path.parent.name
    )

    instruction = (
        data.get("task_prompt_en")
        or data.get("task_prompt")
        or data.get("task_prompt_zh")
    )

    if not instruction:
        raise ValueError(
            f"ScreenAgent file has no task prompt: {path}"
        )

    saved_image_name = data.get("saved_image_name")

    screenshot = None

    if saved_image_name:
        screenshot = str(
            path.parent
            / "images"
            / saved_image_name
        )

    raw_actions = data.get("actions") or []

    task_id = f"{session_id}:{path.stem}"

    is_negative_plan = (
        "neg_plan" in path.stem.lower()
    )

    examples: list[GUIExample] = []
    history: list[GUIAction] = []

    for step_index, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            continue

        action = parse_screenagent_action(
            raw_action
        )

        example = GUIExample(
            source="screenagent",
            task_id=task_id,
            instruction=str(instruction),
            step_index=step_index,
            screenshot=screenshot,
            action=action,
            history=tuple(history),
            metadata={
                "session_id": session_id,
                "video_width": data.get(
                    "video_width"
                ),
                "video_height": data.get(
                    "video_height"
                ),
                "is_negative_plan": is_negative_plan,
                "source_file": str(path),
            },
        )

        examples.append(example)
        history.append(action)

    return examples