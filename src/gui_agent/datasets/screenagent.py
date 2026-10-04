from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gui_agent.datasets.schema import (
    ActionType,
    EvaluationStatus,
    GUIAction,
    GUIExample,
    Point,
)


def _parse_point(
    value: Any,
    *,
    field_name: str = "mouse_position",
    required: bool = False,
) -> Point | None:
    """
    Parse ScreenAgent's:

        {
            "width": x,
            "height": y
        }

    as a pixel-space Point.
    """

    if value is None:
        if required:
            raise ValueError(
                f"{field_name} is required"
            )

        return None

    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} must be an object"
        )

    x = value.get("width")
    y = value.get("height")

    for name, coordinate in (
        ("width", x),
        ("height", y),
    ):
        if (
            isinstance(coordinate, bool)
            or not isinstance(
                coordinate,
                (int, float),
            )
        ):
            raise ValueError(
                f"{field_name}.{name} "
                "must be a finite number"
            )

        if not math.isfinite(
            float(coordinate)
        ):
            raise ValueError(
                f"{field_name}.{name} "
                "must be a finite number"
            )

    return Point(
        x=float(x),
        y=float(y),
    )


def _parse_keyboard_keys(
    value: Any,
) -> tuple[str, ...]:
    """
    Normalize ScreenAgent keyboard_key.

    Official data contains both:

        "Return"
        "2"
        "Ctrl+S"

    and:

        ["Control_L", "s"]
        ["Control_L", "Shift_L", "`"]

    Numeric-looking strings must remain strings.
    """

    if isinstance(value, str):
        key = value.strip()

        if not key:
            raise ValueError(
                "keyboard_key must not be empty"
            )

        # A small number of ScreenAgent samples encode
        # a chord in one string, e.g. "Ctrl+S".
        if "+" in key:
            parts = tuple(
                part.strip()
                for part in key.split("+")
            )

            if (
                len(parts) > 1
                and all(parts)
            ):
                return parts

        return (key,)

    if isinstance(
        value,
        (list, tuple),
    ):
        if not value:
            raise ValueError(
                "keyboard_key sequence "
                "must not be empty"
            )

        keys: list[str] = []

        for index, item in enumerate(
            value
        ):
            if (
                not isinstance(item, str)
                or not item.strip()
            ):
                raise ValueError(
                    "keyboard_key sequence items "
                    "must be non-empty strings "
                    f"(index {index})"
                )

            keys.append(
                item.strip()
            )

        return tuple(keys)

    raise ValueError(
        "keyboard_key must be a string "
        "or a sequence of strings"
    )


def _parse_wait_seconds(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            "wait_time must be a finite "
            "non-negative number"
        )

    seconds = float(value)

    if (
        not math.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError(
            "wait_time must be a finite "
            "non-negative number"
        )

    return seconds


def parse_screenagent_action(
    raw: dict[str, Any],
) -> GUIAction:
    """Convert one raw ScreenAgent action into GUIAction."""

    if not isinstance(raw, dict):
        raise ValueError(
            "ScreenAgent action must be an object"
        )

    raw_type = raw.get(
        "action_type"
    )

    if (
        not isinstance(raw_type, str)
        or not raw_type.strip()
    ):
        raise ValueError(
            "ScreenAgent action_type "
            "must be a non-empty string"
        )

    if raw_type == "PlanAction":
        return GUIAction(
            action_type=ActionType.PLAN,
            element=raw.get(
                "element"
            ),
            raw_action=raw_type,
        )

    if raw_type == "MouseAction":
        mouse_type = raw.get(
            "mouse_action_type"
        )

        if (
            not isinstance(
                mouse_type,
                str,
            )
            or not mouse_type.strip()
        ):
            raise ValueError(
                "MouseAction mouse_action_type "
                "must be a non-empty string"
            )

        action_map = {
            "click": ActionType.CLICK,
            "double_click": (
                ActionType.DOUBLE_CLICK
            ),
            "move": ActionType.MOVE,
            "drag": ActionType.DRAG,
            "down": ActionType.MOUSE_DOWN,
            "up": ActionType.MOUSE_UP,
            "scroll_down": (
                ActionType.SCROLL
            ),
            "scroll_up": (
                ActionType.SCROLL
            ),
        }

        action_type = action_map.get(
            mouse_type,
            ActionType.OTHER,
        )

        # Do not silently discard unknown future
        # ScreenAgent mouse actions.
        if (
            action_type
            == ActionType.OTHER
        ):
            return GUIAction(
                action_type=(
                    ActionType.OTHER
                ),
                raw_action=raw_type,
                metadata=dict(raw),
            )

        is_scroll = mouse_type in {
            "scroll_down",
            "scroll_up",
        }

        position = _parse_point(
            raw.get(
                "mouse_position"
            ),
            required=not is_scroll,
        )

        scroll_delta = None

        if is_scroll:
            repeat = raw.get(
                "scroll_repeat"
            )

            if (
                type(repeat) is not int
                or repeat <= 0
            ):
                raise ValueError(
                    "scroll_repeat must be "
                    "a positive integer"
                )

            if (
                mouse_type
                == "scroll_down"
            ):
                scroll_delta = -repeat
            else:
                scroll_delta = repeat

        metadata = {
            "mouse_action_type": (
                mouse_type
            ),
            "scroll_repeat": raw.get(
                "scroll_repeat"
            ),
        }

        if (
            action_type
            == ActionType.DRAG
        ):
            # Official ScreenAgent samples contain
            # only one coordinate for DRAG.
            #
            # A complete executable drag requires
            # both a start and an end position.
            #
            # Do not invent a missing start point.
            return GUIAction(
                action_type=(
                    ActionType.DRAG
                ),
                position=None,
                end_position=position,
                mouse_button=(
                    raw.get(
                        "mouse_button"
                    )
                    or None
                ),
                raw_action=raw_type,
                metadata={
                    **metadata,
                    "drag_start_available": (
                        False
                    ),
                },
            )

        return GUIAction(
            action_type=action_type,
            position=position,
            mouse_button=(
                raw.get(
                    "mouse_button"
                )
                or None
            ),
            scroll_delta=scroll_delta,
            raw_action=raw_type,
            metadata=metadata,
        )

    if raw_type == "KeyboardAction":
        keyboard_type = raw.get(
            "keyboard_action_type"
        )

        if (
            not isinstance(
                keyboard_type,
                str,
            )
            or not keyboard_type.strip()
        ):
            raise ValueError(
                "KeyboardAction "
                "keyboard_action_type "
                "must be a non-empty string"
            )

        if keyboard_type == "text":
            text = raw.get(
                "keyboard_text"
            )

            if not isinstance(
                text,
                str,
            ):
                raise ValueError(
                    "keyboard_text "
                    "must be a string"
                )

            return GUIAction(
                action_type=(
                    ActionType.TYPE_TEXT
                ),
                text=text,
                raw_action=raw_type,
                metadata={
                    "keyboard_action_type": (
                        keyboard_type
                    ),
                },
            )

        if keyboard_type == "press":
            keys = (
                _parse_keyboard_keys(
                    raw.get(
                        "keyboard_key"
                    )
                )
            )

            if len(keys) == 1:
                action_type = (
                    ActionType.KEY_PRESS
                )
            else:
                action_type = (
                    ActionType.HOTKEY
                )

            return GUIAction(
                action_type=(
                    action_type
                ),
                keys=keys,
                raw_action=raw_type,
                metadata={
                    "keyboard_action_type": (
                        keyboard_type
                    ),
                    "raw_keyboard_key": (
                        raw.get(
                            "keyboard_key"
                        )
                    ),
                },
            )

        return GUIAction(
            action_type=(
                ActionType.OTHER
            ),
            raw_action=raw_type,
            metadata=dict(raw),
        )

    if raw_type == "WaitAction":
        return GUIAction(
            action_type=ActionType.WAIT,
            wait_seconds=(
                _parse_wait_seconds(
                    raw.get(
                        "wait_time"
                    )
                )
            ),
            raw_action=raw_type,
        )

    if (
        raw_type
        == "EvaluateSubTaskAction"
    ):
        raw_situation = raw.get(
            "situation"
        )

        try:
            status = (
                EvaluationStatus(
                    raw_situation
                )
            )
        except (
            ValueError,
            TypeError,
        ):
            status = None

        return GUIAction(
            action_type=(
                ActionType.EVALUATE
            ),
            evaluation_status=status,
            advice=raw.get(
                "advice"
            ),
            raw_action=raw_type,
            metadata={
                "raw_situation": (
                    raw_situation
                ),
            },
        )

    # Unknown future top-level action types
    # remain visible in the dataset.
    return GUIAction(
        action_type=(
            ActionType.OTHER
        ),
        raw_action=raw_type,
        metadata=dict(raw),
    )


def _first_nonempty_text(
    data: dict[str, Any],
    *field_names: str,
) -> str | None:
    for field_name in field_names:
        value = data.get(
            field_name
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return value

    return None


def load_screenagent_file(
    path: str | Path,
) -> list[GUIExample]:
    """
    Load one ScreenAgent JSON file.

    One ScreenAgent JSON may contain multiple actions,
    therefore one GUIExample is returned per action.

    ``actions=[]`` is valid.

    Malformed containers or action items fail explicitly
    rather than being silently dropped.
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "ScreenAgent file must "
            "contain a JSON object: "
            f"{path}"
        )

    session_id = data.get(
        "session_id"
    )

    if (
        not isinstance(
            session_id,
            str,
        )
        or not session_id.strip()
    ):
        raise ValueError(
            "ScreenAgent file has "
            "invalid session_id: "
            f"{path}"
        )

    instruction = (
        _first_nonempty_text(
            data,
            "task_prompt_en",
            "task_prompt",
            "task_prompt_zh",
        )
    )

    if instruction is None:
        raise ValueError(
            "ScreenAgent file has "
            "no task prompt: "
            f"{path}"
        )

    saved_image_name = data.get(
        "saved_image_name"
    )

    if (
        saved_image_name
        is not None
        and (
            not isinstance(
                saved_image_name,
                str,
            )
            or not (
                saved_image_name.strip()
            )
        )
    ):
        raise ValueError(
            "ScreenAgent file has "
            "invalid saved_image_name: "
            f"{path}"
        )

    screenshot = None

    if saved_image_name:
        screenshot = str(
            path.parent
            / "images"
            / saved_image_name
        )

    if "actions" not in data:
        raise ValueError(
            "ScreenAgent file "
            "is missing actions: "
            f"{path}"
        )

    raw_actions = data[
        "actions"
    ]

    if raw_actions is None:
        raise ValueError(
            "ScreenAgent actions "
            "must not be null: "
            f"{path}"
        )

    if not isinstance(
        raw_actions,
        list,
    ):
        raise ValueError(
            "ScreenAgent actions "
            "must be a list: "
            f"{path}"
        )

    task_id = (
        f"{session_id}:{path.stem}"
    )

    is_negative_plan = (
        "neg_plan"
        in path.stem.lower()
    )

    examples: list[
        GUIExample
    ] = []

    history: list[
        GUIAction
    ] = []

    for (
        step_index,
        raw_action,
    ) in enumerate(
        raw_actions
    ):
        if not isinstance(
            raw_action,
            dict,
        ):
            raise ValueError(
                "ScreenAgent action item "
                "must be an object: "
                f"{path} "
                f"action_index={step_index}"
            )

        try:
            action = (
                parse_screenagent_action(
                    raw_action
                )
            )

        except ValueError as exc:
            raise ValueError(
                "Invalid ScreenAgent action: "
                f"{path} "
                f"action_index={step_index}: "
                f"{exc}"
            ) from exc

        example = GUIExample(
            source="screenagent",
            task_id=task_id,
            instruction=instruction,
            step_index=step_index,
            screenshot=screenshot,
            action=action,
            history=tuple(
                history
            ),
            metadata={
                "session_id": (
                    session_id
                ),
                "video_width": (
                    data.get(
                        "video_width"
                    )
                ),
                "video_height": (
                    data.get(
                        "video_height"
                    )
                ),
                "is_negative_plan": (
                    is_negative_plan
                ),
                "source_file": str(
                    path
                ),
            },
        )

        examples.append(
            example
        )

        history.append(
            action
        )

    return examples