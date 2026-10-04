from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from gui_agent.datasets.schema import ActionType, GUIAction


PLANNER_ACTION_TYPES = frozenset(
    {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.TYPE_TEXT,
        ActionType.KEY_PRESS,
        ActionType.HOTKEY,
        ActionType.SCROLL,
        ActionType.WAIT,
    }
)

_PLAN_FIELDS = frozenset({"steps"})
_STEP_FIELDS = frozenset(
    {
        "step_id",
        "description",
        "action",
    }
)

_ACTION_FIELDS = {
    ActionType.CLICK: frozenset(
        {
            "action_type",
            "element",
        }
    ),
    ActionType.DOUBLE_CLICK: frozenset(
        {
            "action_type",
            "element",
        }
    ),
    ActionType.TYPE_TEXT: frozenset(
        {
            "action_type",
            "text",
        }
    ),
    ActionType.KEY_PRESS: frozenset(
        {
            "action_type",
            "keys",
        }
    ),
    ActionType.HOTKEY: frozenset(
        {
            "action_type",
            "keys",
        }
    ),
    ActionType.SCROLL: frozenset(
        {
            "action_type",
            "scroll_delta",
        }
    ),
    ActionType.WAIT: frozenset(
        {
            "action_type",
            "wait_seconds",
        }
    ),
}


def _reject_unknown_fields(
    data: dict[str, Any],
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = set(data) - allowed

    if unknown:
        names = ", ".join(
            sorted(unknown)
        )

        raise ValueError(
            f"{context} contains unknown field(s): "
            f"{names}"
        )


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value


def _parse_keys(
    data: dict[str, Any],
    *,
    exact_count: int | None = None,
) -> tuple[str, ...]:
    raw_keys = data.get("keys")

    if not isinstance(
        raw_keys,
        (list, tuple),
    ):
        raise ValueError(
            "keys must be a list"
        )

    if not raw_keys:
        raise ValueError(
            "keys must not be empty"
        )

    if not all(
        isinstance(key, str)
        and key.strip()
        for key in raw_keys
    ):
        raise ValueError(
            "keys must contain non-empty strings"
        )

    if (
        exact_count is not None
        and len(raw_keys) != exact_count
    ):
        raise ValueError(
            "keys must contain exactly "
            f"{exact_count} item(s)"
        )

    return tuple(raw_keys)


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
            "wait_seconds must be a number"
        )

    wait_seconds = float(value)

    if not math.isfinite(
        wait_seconds
    ):
        raise ValueError(
            "wait_seconds must be finite"
        )

    if wait_seconds < 0:
        raise ValueError(
            "wait_seconds must be non-negative"
        )

    return wait_seconds


@dataclass(frozen=True)
class PlanStep:
    step_id: int
    description: str
    action: GUIAction | None = None

    def __post_init__(
        self,
    ) -> None:
        if (
            type(self.step_id) is not int
            or self.step_id < 1
        ):
            raise ValueError(
                "step_id must be a positive integer"
            )

        if (
            not isinstance(
                self.description,
                str,
            )
            or not self.description.strip()
        ):
            raise ValueError(
                "description must not be empty"
            )


def _parse_gui_action(
    data: Any,
) -> GUIAction | None:
    """
    Parse one planner action using the strict
    planner-output contract.

    GUIAction itself remains a broad dataset/runtime
    representation. Planner output is intentionally
    narrower.
    """

    if data is None:
        # Preserve compatibility with legacy
        # high-level plans without actions.
        return None

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "action must be an object"
        )

    raw_type = data.get(
        "action_type"
    )

    try:
        action_type = ActionType(
            raw_type
        )

    except (
        ValueError,
        TypeError,
    ) as exc:
        raise ValueError(
            f"invalid action_type: {raw_type}"
        ) from exc

    if (
        action_type
        not in PLANNER_ACTION_TYPES
    ):
        raise ValueError(
            "action_type is not allowed "
            "in planner output: "
            f"{action_type.value}"
        )

    _reject_unknown_fields(
        data,
        allowed=_ACTION_FIELDS[
            action_type
        ],
        context=(
            f"{action_type.value} action"
        ),
    )

    if action_type in {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
    }:
        element = (
            _require_non_empty_string(
                data.get("element"),
                field_name="element",
            )
        )

        return GUIAction(
            action_type=action_type,
            element=element,
        )

    if (
        action_type
        == ActionType.TYPE_TEXT
    ):
        text = data.get("text")

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                "text must be a string"
            )

        return GUIAction(
            action_type=action_type,
            text=text,
        )

    if (
        action_type
        == ActionType.KEY_PRESS
    ):
        return GUIAction(
            action_type=action_type,
            keys=_parse_keys(
                data,
                exact_count=1,
            ),
        )

    if (
        action_type
        == ActionType.HOTKEY
    ):
        return GUIAction(
            action_type=action_type,
            keys=_parse_keys(
                data
            ),
        )

    if (
        action_type
        == ActionType.SCROLL
    ):
        scroll_delta = data.get(
            "scroll_delta"
        )

        # bool is a subclass of int,
        # so use exact type checking here.
        if type(scroll_delta) is not int:
            raise ValueError(
                "scroll_delta must be an integer"
            )

        return GUIAction(
            action_type=action_type,
            scroll_delta=scroll_delta,
        )

    if (
        action_type
        == ActionType.WAIT
    ):
        return GUIAction(
            action_type=action_type,
            wait_seconds=(
                _parse_wait_seconds(
                    data.get(
                        "wait_seconds"
                    )
                )
            ),
        )

    # Defensive protection in case
    # PLANNER_ACTION_TYPES is extended without
    # adding a parser branch above.
    raise ValueError(
        "unsupported planner action_type: "
        f"{action_type.value}"
    )


@dataclass(frozen=True)
class TaskPlan:
    """Validated high-level task plan."""

    steps: tuple[PlanStep, ...]

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.steps,
            tuple,
        ):
            raise ValueError(
                "steps must be a tuple"
            )

        if not 1 <= len(
            self.steps
        ) <= 8:
            raise ValueError(
                "plan must contain between "
                "1 and 8 steps"
            )

        for (
            expected_id,
            step,
        ) in enumerate(
            self.steps,
            start=1,
        ):
            if not isinstance(
                step,
                PlanStep,
            ):
                raise ValueError(
                    "Every step must be a PlanStep"
                )

            if (
                step.step_id
                != expected_id
            ):
                raise ValueError(
                    "step_id must be consecutive "
                    "and start from 1"
                )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> TaskPlan:
        """
        Build a validated task plan from
        model-generated dictionary data.
        """

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Plan must be a JSON object"
            )

        _reject_unknown_fields(
            data,
            allowed=_PLAN_FIELDS,
            context="plan",
        )

        raw_steps = data.get(
            "steps"
        )

        if not isinstance(
            raw_steps,
            list,
        ):
            raise ValueError(
                "steps must be a list"
            )

        steps: list[
            PlanStep
        ] = []

        for item in raw_steps:
            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    "Each step must be an object"
                )

            _reject_unknown_fields(
                item,
                allowed=_STEP_FIELDS,
                context="plan step",
            )

            steps.append(
                PlanStep(
                    step_id=item.get(
                        "step_id"
                    ),
                    description=item.get(
                        "description"
                    ),
                    action=_parse_gui_action(
                        item.get(
                            "action"
                        )
                    ),
                )
            )

        return cls(
            steps=tuple(
                steps
            )
        )

    @classmethod
    def from_json(
        cls,
        text: str,
    ) -> TaskPlan:
        """
        Parse and validate a model JSON response.
        """

        if not isinstance(
            text,
            str,
        ):
            raise ValueError(
                "Model response must be a string"
            )

        data = json.loads(
            text
        )

        return cls.from_dict(
            data
        )