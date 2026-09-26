from __future__ import annotations

import time
from typing import Any

from gui_agent.agent.action_adapter import validate_executable_action
from gui_agent.datasets.schema import ActionType, GUIAction


class ActionExecutionError(RuntimeError):
    pass


class ActionExecutor:
    def __init__(self, controller: Any):
        self.controller = controller

    def execute(self, action: GUIAction) -> None:
        validate_executable_action(action)

        try:
            action_type = action.action_type

            if action_type == ActionType.CLICK:
                self.controller.click(
                    (action.position.x, action.position.y),
                    button=action.mouse_button or "left",
                )

            elif action_type == ActionType.DOUBLE_CLICK:
                self.controller.double_click(
                    (action.position.x, action.position.y),
                    button=action.mouse_button or "left",
                )

            elif action_type == ActionType.MOVE:
                self.controller.move_to(
                    (action.position.x, action.position.y)
                )

            elif action_type == ActionType.DRAG:
                self.controller.move_to(
                    (action.position.x, action.position.y)
                )
                self.controller.drag_to(
                    (
                        action.end_position.x,
                        action.end_position.y,
                    ),
                    button=action.mouse_button or "left",
                )

            elif action_type == ActionType.SCROLL:
                self.controller.scroll(action.scroll_delta)

            elif action_type == ActionType.TYPE_TEXT:
                self.controller.type_text(action.text)

            elif action_type == ActionType.KEY_PRESS:
                self.controller.press(action.keys[0])

            elif action_type == ActionType.HOTKEY:
                self.controller.hotkey(*action.keys)

            elif action_type == ActionType.WAIT:
                time.sleep(action.wait_seconds)

            else:
                raise ActionExecutionError(
                    f"unsupported action type: {action_type.value}"
                )

        except Exception as exc:
            if isinstance(exc, ActionExecutionError):
                raise

            raise ActionExecutionError(
                f"failed to execute {action.action_type.value}: {exc}"
            ) from exc