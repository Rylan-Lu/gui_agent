from __future__ import annotations

from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.grounding import GroundingBridge
from gui_agent.agent.observation import Observation
from gui_agent.datasets.schema import ActionType, GUIAction


class ActionRunnerError(RuntimeError):
    pass


class ActionRunner:
    def __init__(
        self,
        executor: ActionExecutor,
        grounding: GroundingBridge | None = None,
    ):
        self.executor = executor
        self.grounding = grounding or GroundingBridge()

    def run(
        self,
        action: GUIAction,
        observation: Observation,
    ) -> GUIAction:

        resolved_action = action

        if (
            action.action_type
            in {ActionType.CLICK, ActionType.DOUBLE_CLICK}
            and action.position is None
        ):
            if observation.ocr_result is None:
                raise ActionRunnerError(
                    "grounding requires OCR results"
                )

            image_size = observation.metadata.get("image_size")
            region = observation.metadata.get("region")

            if image_size is None or region is None:
                raise ActionRunnerError(
                    "observation is missing image_size or region"
                )

            resolved_action = self.grounding.ground(
                action,
                ocr_results=observation.ocr_result,
                image_size=image_size,
                region=region,
            )

        self.executor.execute(resolved_action)

        return resolved_action