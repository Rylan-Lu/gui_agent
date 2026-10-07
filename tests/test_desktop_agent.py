from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gui_agent.agent.action_transaction import ActionTransactionResult
from gui_agent.agent.desktop_agent import DesktopAgent
from gui_agent.agent.feedback import FeedbackResult
from gui_agent.agent.observation import Observation
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.success_judge import SuccessJudgeResult


def _observation(value: int = 0) -> Observation:
    return Observation(
        screenshot=np.full(
            (12, 20, 3),
            value,
            dtype=np.uint8,
        ),
        screen_width=20,
        screen_height=12,
        metadata={
            "region": (0, 0, 20, 12),
            "image_size": (20, 12),
        },
    )


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def observe(self, *, use_ocr: bool = False) -> Observation:
        self.calls.append(use_ocr)
        return _observation()


class RecordingPlanner:
    def __init__(self) -> None:
        self.instruction: str | None = None
        self.screenshot_path: Path | None = None
        self.decoded_shape: tuple[int, ...] | None = None

    def plan(
        self,
        instruction: str,
        screenshot_path: Path | None = None,
    ) -> TaskPlan:
        assert screenshot_path is not None
        assert screenshot_path.is_file()

        decoded = cv2.imread(str(screenshot_path))
        assert decoded is not None

        self.instruction = instruction
        self.screenshot_path = screenshot_path
        self.decoded_shape = decoded.shape

        return TaskPlan.from_dict(
            {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Wait briefly",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 0.0,
                        },
                    }
                ]
            }
        )


class FakeTransaction:
    def __init__(self) -> None:
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        before = _observation(0)
        after = _observation(1)
        return ActionTransactionResult(
            action=action,
            before=before,
            after=after,
            feedback=FeedbackResult(
                changed=True,
                changed_pixel_ratio=1.0,
                mean_abs_diff=1.0,
            ),
        )


def test_run_captures_live_screen_plans_and_executes() -> None:
    environment = FakeEnvironment()
    planner = RecordingPlanner()
    transaction = FakeTransaction()

    agent = DesktopAgent(
        planner=planner,
        environment=environment,
        transaction=transaction,
    )

    result = agent.run("  do the visible task  ")

    assert result.success is True
    assert result.instruction == "do the visible task"
    assert planner.instruction == "do the visible task"
    assert planner.decoded_shape == (12, 20, 3)
    assert environment.calls == [False]
    assert len(transaction.actions) == 1
    assert len(result.execution.transactions) == 1

    assert planner.screenshot_path is not None
    assert not planner.screenshot_path.exists()

    assert result.capture_seconds >= 0
    assert result.planning_seconds >= 0
    assert result.execution_seconds >= 0
    assert result.total_seconds >= 0


def test_run_forwards_final_check() -> None:
    environment = FakeEnvironment()
    planner = RecordingPlanner()
    transaction = FakeTransaction()

    agent = DesktopAgent(
        planner=planner,
        environment=environment,
        transaction=transaction,
    )

    result = agent.run(
        "do task",
        final_check=lambda: SuccessJudgeResult(
            success=False,
            reason="expected state missing",
        ),
    )

    assert result.success is False
    assert result.execution.failure_reason == "expected state missing"
    assert result.execution.judge_result is not None
    assert result.execution.judge_result.success is False


@pytest.mark.parametrize(
    "instruction",
    ["", "   ", None],
)
def test_run_rejects_empty_instruction(instruction) -> None:
    environment = FakeEnvironment()
    planner = RecordingPlanner()
    transaction = FakeTransaction()

    agent = DesktopAgent(
        planner=planner,
        environment=environment,
        transaction=transaction,
    )

    with pytest.raises(ValueError, match="instruction must not be empty"):
        agent.run(instruction)

    assert environment.calls == []
    assert planner.instruction is None
    assert transaction.actions == []


@pytest.mark.parametrize(
    ("max_steps", "timeout_s"),
    [
        (0, 60.0),
        (True, 60.0),
        (10, 0.0),
        (10, float("inf")),
        (10, True),
    ],
)
def test_constructor_rejects_invalid_limits(
    max_steps,
    timeout_s,
) -> None:
    with pytest.raises(ValueError):
        DesktopAgent(
            planner=RecordingPlanner(),
            environment=FakeEnvironment(),
            transaction=FakeTransaction(),
            max_steps=max_steps,
            timeout_s=timeout_s,
        )


class _UnloadableModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.unload_calls = 0

    def unload(self) -> None:
        self.unload_calls += 1
        self.events.append("unload")


class _PlannerWithUnloadableModel(RecordingPlanner):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.model = _UnloadableModel(events)
        self.events = events

    def plan(
        self,
        instruction: str,
        screenshot_path: Path | None = None,
    ) -> TaskPlan:
        self.events.append("plan")
        return super().plan(instruction, screenshot_path)


class _OrderedTransaction(FakeTransaction):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def execute(self, action):
        self.events.append("execute")
        return super().execute(action)


def test_run_releases_planner_model_before_execution() -> None:
    events: list[str] = []
    planner = _PlannerWithUnloadableModel(events)

    agent = DesktopAgent(
        planner=planner,
        environment=FakeEnvironment(),
        transaction=_OrderedTransaction(events),
    )

    result = agent.run("do task")

    assert result.success is True
    assert planner.model.unload_calls == 1
    assert events == ["plan", "unload", "execute"]


def test_run_can_keep_planner_model_loaded() -> None:
    events: list[str] = []
    planner = _PlannerWithUnloadableModel(events)

    agent = DesktopAgent(
        planner=planner,
        environment=FakeEnvironment(),
        transaction=_OrderedTransaction(events),
        release_planner_after_plan=False,
    )

    result = agent.run("do task")

    assert result.success is True
    assert planner.model.unload_calls == 0
    assert events == ["plan", "execute"]


def test_constructor_rejects_invalid_release_flag() -> None:
    with pytest.raises(
        ValueError,
        match="release_planner_after_plan must be a boolean",
    ):
        DesktopAgent(
            planner=RecordingPlanner(),
            environment=FakeEnvironment(),
            transaction=FakeTransaction(),
            release_planner_after_plan=1,
        )
