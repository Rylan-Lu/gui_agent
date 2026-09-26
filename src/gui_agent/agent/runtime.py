from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from gui_agent.agent.observation import Observation


@dataclass(slots=True)
class StepRecord:
    """
    Record for one Agent execution step.

    This intentionally does not define a new action schema.
    Executable GUI actions will continue to reuse the project's
    existing GUIAction / ActionType definitions.
    """

    step_index: int
    description: str

    success: bool | None = None
    error: str | None = None


@dataclass(slots=True)
class AgentState:
    """
    Runtime state of one GUI Agent task.
    """

    task: str

    max_steps: int = 10
    timeout_s: float = 60.0

    current_step: int = 0
    step_count: int = 0

    history: list[StepRecord] = field(default_factory=list)
    observation: Observation | None = None

    started_at: float | None = None

    done: bool = False
    failed: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")

        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than 0")


class AgentRuntime:
    """
    Controls the lifecycle of one GUI Agent task.

    Responsibilities:
    - hold task state
    - track observations
    - track execution steps
    - enforce max_steps
    - enforce timeout
    - mark task success/failure
    """

    def __init__(self, state: AgentState):
        self.state = state

    @classmethod
    def create(
        cls,
        task: str,
        *,
        max_steps: int = 10,
        timeout_s: float = 60.0,
    ) -> "AgentRuntime":
        state = AgentState(
            task=task,
            max_steps=max_steps,
            timeout_s=timeout_s,
        )

        return cls(state)

    def start(self) -> AgentState:
        if self.state.started_at is None:
            self.state.started_at = time.monotonic()

        return self.state

    @property
    def elapsed_s(self) -> float:
        if self.state.started_at is None:
            return 0.0

        return time.monotonic() - self.state.started_at

    def update_observation(self, observation: Any) -> None:
        self.state.observation = observation

    def begin_step(self, description: str) -> StepRecord:
        if not description.strip():
            raise ValueError("step description must not be empty")

        if not self.check_limits():
            raise RuntimeError(
                self.state.failure_reason or "agent cannot continue"
            )

        self.state.step_count += 1
        self.state.current_step = self.state.step_count

        record = StepRecord(
            step_index=self.state.current_step,
            description=description,
        )

        self.state.history.append(record)

        return record

    def finish_step(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> StepRecord:
        if not self.state.history:
            raise RuntimeError("no active step to finish")

        record = self.state.history[-1]

        record.success = success
        record.error = error

        return record

    def mark_done(self) -> None:
        self.state.done = True
        self.state.failed = False
        self.state.failure_reason = None

    def mark_failed(self, reason: str) -> None:
        self.state.done = False
        self.state.failed = True
        self.state.failure_reason = reason

    def check_limits(self) -> bool:
        """
        Return True if the Agent may execute another step.
        """

        if self.state.done or self.state.failed:
            return False

        if self.state.started_at is None:
            self.start()

        if self.state.step_count >= self.state.max_steps:
            self.mark_failed(
                f"max_steps exceeded: {self.state.max_steps}"
            )
            return False

        if self.elapsed_s >= self.state.timeout_s:
            self.mark_failed(
                f"timeout exceeded: {self.state.timeout_s:.2f}s"
            )
            return False

        return True