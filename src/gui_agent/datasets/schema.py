from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Normalized GUI / agent action types."""

    # High-level reasoning
    PLAN = "plan"
    EVALUATE = "evaluate"

    # Mouse
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    MOVE = "move"
    DRAG = "drag"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    SCROLL = "scroll"
    SELECT = "select"

    # Keyboard
    TYPE_TEXT = "type_text"
    KEY_PRESS = "key_press"
    HOTKEY = "hotkey"

    # Runtime
    WAIT = "wait"

    # Fallback for unknown dataset-specific actions
    OTHER = "other"


class CoordinateSpace(str, Enum):
    """Coordinate representation used by a GUI action."""

    PIXEL = "pixel"
    NORMALIZED = "normalized"


class EvaluationStatus(str, Enum):
    """Normalized result of evaluating a GUI sub-task."""

    SUB_TASK_SUCCESS = "sub_task_success"
    NEED_RETRY = "need_retry"
    NEED_REFORMULATE = "need_reformulate"


@dataclass(frozen=True)
class Point:
    """A 2D GUI coordinate."""

    x: float
    y: float


@dataclass(frozen=True)
class GUIAction:
    """
    Unified action representation used across GUI datasets.

    Dataset-specific information that does not yet have a common
    representation is preserved in ``metadata``.
    """

    action_type: ActionType

    # Mouse coordinates
    position: Point | None = None
    end_position: Point | None = None
    coordinate_space: CoordinateSpace = CoordinateSpace.PIXEL

    # Mouse properties
    mouse_button: str | None = None
    scroll_delta: int | None = None

    # Keyboard properties
    text: str | None = None
    keys: tuple[str, ...] = ()

    # Planning
    element: str | None = None

    # Waiting
    wait_seconds: float | None = None

    # Evaluation
    evaluation_status: EvaluationStatus | None = None
    advice: str | None = None

    # Original dataset information
    raw_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GUIExample:
    """
    Unified representation of one GUI dataset sample / trajectory step.
    """

    source: str
    task_id: str
    instruction: str

    step_index: int = 0
    screenshot: str | None = None

    action: GUIAction | None = None
    history: tuple[GUIAction, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must not be empty")

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")

        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")

        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")