from __future__ import annotations

from gui_agent.datasets.schema import ActionType, CoordinateSpace, GUIAction
from gui_agent.agent.plan_schema import PlanStep


class ActionValidationError(ValueError):
    pass


EXECUTABLE_ACTION_TYPES = {
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.MOVE,
    ActionType.DRAG,
    ActionType.SCROLL,
    ActionType.TYPE_TEXT,
    ActionType.KEY_PRESS,
    ActionType.HOTKEY,
    ActionType.WAIT,
}

def plan_step_to_action(step: PlanStep) -> GUIAction:
    if step.action is None:
        raise ActionValidationError(
            f"plan step {step.step_id} has no action"
        )

    return step.action


def is_executable_action(action: GUIAction) -> bool:
    return action.action_type in EXECUTABLE_ACTION_TYPES


def validate_executable_action(action: GUIAction) -> GUIAction:
    if not is_executable_action(action):
        raise ActionValidationError(
            f"action type is not directly executable: {action.action_type.value}"
        )

    if action.coordinate_space != CoordinateSpace.PIXEL:
        raise ActionValidationError(
            "only pixel coordinates are executable in Week 4 v1"
        )

    action_type = action.action_type

    if action_type in {
        ActionType.CLICK,
        ActionType.DOUBLE_CLICK,
        ActionType.MOVE,
    }:
        if action.position is None:
            raise ActionValidationError(
                f"{action_type.value} requires position"
            )

    elif action_type == ActionType.DRAG:
        if action.position is None:
            raise ActionValidationError("drag requires start position")

        if action.end_position is None:
            raise ActionValidationError("drag requires end position")

    elif action_type == ActionType.SCROLL:
        if action.scroll_delta is None:
            raise ActionValidationError("scroll requires scroll_delta")

    elif action_type == ActionType.TYPE_TEXT:
        if action.text is None:
            raise ActionValidationError("type_text requires text")

    elif action_type == ActionType.KEY_PRESS:
        if len(action.keys) != 1:
            raise ActionValidationError(
                "key_press requires exactly one key"
            )

    elif action_type == ActionType.HOTKEY:
        if not action.keys:
            raise ActionValidationError(
                "hotkey requires at least one key"
            )

    elif action_type == ActionType.WAIT:
        if action.wait_seconds is None:
            raise ActionValidationError(
                "wait requires wait_seconds"
            )

        if action.wait_seconds < 0:
            raise ActionValidationError(
                "wait_seconds must be non-negative"
            )

    if action.mouse_button not in {None, "left", "right", "middle"}:
        raise ActionValidationError(
            f"unsupported mouse button: {action.mouse_button}"
        )

    return action