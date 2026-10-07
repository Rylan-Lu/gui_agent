from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.environment import DesktopEnvironment
from gui_agent.agent.loop import AgentLoop, AgentLoopResult
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.planner import Planner
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import SuccessJudgeResult

FinalCheck = Callable[[], SuccessJudgeResult]


@dataclass(frozen=True, slots=True)
class DesktopAgentResult:
    """Result of one natural-language desktop-agent run."""

    instruction: str
    plan: TaskPlan
    execution: AgentLoopResult
    capture_seconds: float
    planning_seconds: float
    execution_seconds: float

    @property
    def success(self) -> bool:
        return self.execution.success

    @property
    def total_seconds(self) -> float:
        return (
            self.capture_seconds
            + self.planning_seconds
            + self.execution_seconds
        )


class DesktopAgent:
    """
    Thin application layer joining planning and desktop execution.

    The class deliberately reuses the existing Planner, AgentRuntime,
    AgentLoop, DesktopEnvironment, and ActionTransaction contracts instead
    of duplicating their responsibilities.
    """

    def __init__(
        self,
        planner: Planner,
        environment: DesktopEnvironment,
        transaction: ActionTransaction,
        *,
        max_steps: int = 10,
        timeout_s: float = 60.0,
        release_planner_after_plan: bool = True,
    ) -> None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
        ):
            raise ValueError("timeout_s must be a number")

        timeout_s = float(timeout_s)

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(
                "timeout_s must be finite and greater than 0"
            )

        self.planner = planner
        self.environment = environment
        self.transaction = transaction
        if type(release_planner_after_plan) is not bool:
            raise ValueError(
                "release_planner_after_plan must be a boolean"
            )

        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.release_planner_after_plan = release_planner_after_plan

    @staticmethod
    def _validate_instruction(instruction: str) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must not be empty")
        return instruction.strip()


    def _release_planner_resources(self) -> None:
        """Release optional heavyweight planner-model resources."""

        if not self.release_planner_after_plan:
            return

        model = getattr(self.planner, "model", None)
        unload = getattr(model, "unload", None)

        if callable(unload):
            unload()

    @staticmethod
    def _write_planner_screenshot(
        screenshot,
        path: Path,
    ) -> None:
        if screenshot is None:
            raise RuntimeError("desktop observation has no screenshot")

        written = cv2.imwrite(
            str(path),
            screenshot,
        )

        if not written:
            raise RuntimeError(
                f"failed to write planner screenshot: {path}"
            )

    def run(
        self,
        instruction: str,
        *,
        final_check: FinalCheck | None = None,
    ) -> DesktopAgentResult:
        """
        Capture the live desktop, plan from that screenshot, and execute.

        The AgentRuntime budget starts when desktop execution begins. Capture
        and planning timings are reported separately so model latency is not
        confused with the existing execution timeout contract.
        """

        instruction = self._validate_instruction(instruction)

        capture_started = time.perf_counter()
        observation = self.environment.observe(use_ocr=False)
        capture_seconds = time.perf_counter() - capture_started

        with tempfile.TemporaryDirectory(
            prefix="gui_agent_planner_"
        ) as directory:
            screenshot_path = Path(directory) / "desktop.png"
            self._write_planner_screenshot(
                observation.screenshot,
                screenshot_path,
            )

            planning_started = time.perf_counter()
            try:
                plan = self.planner.plan(
                    instruction=instruction,
                    screenshot_path=screenshot_path,
                )
            finally:
                self._release_planner_resources()

            # The planning phase includes optional model-resource release so
            # total_seconds remains a complete wall-clock accounting.
            planning_seconds = time.perf_counter() - planning_started

        runtime = AgentRuntime.create(
            task=instruction,
            max_steps=self.max_steps,
            timeout_s=self.timeout_s,
        )
        loop = AgentLoop(
            runtime,
            self.transaction,
        )

        execution_started = time.perf_counter()
        execution = loop.run(
            plan,
            final_check=final_check,
        )
        execution_seconds = time.perf_counter() - execution_started

        return DesktopAgentResult(
            instruction=instruction,
            plan=plan,
            execution=execution,
            capture_seconds=capture_seconds,
            planning_seconds=planning_seconds,
            execution_seconds=execution_seconds,
        )
