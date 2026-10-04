from __future__ import annotations

from dataclasses import dataclass

from gui_agent.agent.action_adapter import (
    ActionValidationError,
    validate_executable_action,
)
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.grounding import (
    GROUNDABLE_ACTION_TYPES,
    GroundingBridge,
)
from gui_agent.agent.observation import Observation
from gui_agent.datasets.schema import GUIAction


class ActionRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionRunRequirements:
    """
    Requirements that must be satisfied before an action can run.

    At present, OCR is the only expensive optional dependency that
    ActionTransaction needs to know about.
    """

    needs_ocr: bool = False


_SUPPORTED_MOUSE_BUTTONS = {
    None,
    "left",
    "right",
    "middle",
}


class ActionRunner:
    def __init__(
        self,
        executor: ActionExecutor,
        grounding: GroundingBridge | None = None,
    ):
        self.executor = executor
        self.grounding = grounding or GroundingBridge()

    def preflight(
        self,
        action: GUIAction,
    ) -> ActionRunRequirements:
        """
        Validate an action before any screenshot/OCR work begins.

        There are two valid states:

        1. The action is already directly executable.
        2. The action is groundable, but still needs OCR to resolve
           an element into a pixel position.

        This method intentionally performs no screenshot, OCR,
        grounding, or desktop side effect.
        """

        if not isinstance(action, GUIAction):
            raise ActionValidationError(
                "action must be a GUIAction"
            )

        needs_grounding = (
            action.action_type in GROUNDABLE_ACTION_TYPES
            and action.position is None
        )

        if needs_grounding:
            if (
                not isinstance(action.element, str)
                or not action.element.strip()
            ):
                raise ActionValidationError(
                    f"{action.action_type.value} requires "
                    "a non-empty element when position is absent"
                )

            # Validate fields that can already be checked before
            # Grounding. This prevents an invalid action from
            # unnecessarily triggering OCR.
            if (
                action.mouse_button
                not in _SUPPORTED_MOUSE_BUTTONS
            ):
                raise ActionValidationError(
                    "unsupported mouse button: "
                    f"{action.mouse_button}"
                )

            return ActionRunRequirements(
                needs_ocr=True
            )

        # Any action that does not require Grounding must already
        # satisfy the complete executable-action contract.
        validate_executable_action(action)

        return ActionRunRequirements(
            needs_ocr=False
        )

    def run(
        self,
        action: GUIAction,
        observation: Observation,
    ) -> GUIAction:
        """
        Resolve the action if necessary and execute it.

        preflight() is intentionally called again here even when
        ActionTransaction already called it. This keeps ActionRunner
        safe when used directly outside ActionTransaction.
        """

        requirements = self.preflight(
            action
        )

        resolved_action = action

        if requirements.needs_ocr:
            if observation.ocr_result is None:
                raise ActionRunnerError(
                    "grounding requires OCR results"
                )

            image_size = observation.metadata.get(
                "image_size"
            )
            region = observation.metadata.get(
                "region"
            )

            if (
                image_size is None
                or region is None
            ):
                raise ActionRunnerError(
                    "observation is missing "
                    "image_size or region"
                )

            resolved_action = (
                self.grounding.ground(
                    action,
                    ocr_results=(
                        observation.ocr_result
                    ),
                    image_size=image_size,
                    region=region,
                )
            )

        # GroundingError and ActionExecutionError are deliberately
        # not wrapped here. Higher layers need to distinguish:
        #
        #   grounding failure
        #   execution failure
        #
        # from ActionRunner's own observation-contract failures.
        self.executor.execute(
            resolved_action
        )

        return resolved_action