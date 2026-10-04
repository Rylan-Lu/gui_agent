from dataclasses import dataclass

import gui_agent.agent.runtime as runtime_module
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.observation import Observation
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
)


class FakeClock:
    def __init__(
        self,
        now=0.0,
    ):
        self.now = float(now)

    def monotonic(self):
        return self.now

    def advance(
        self,
        seconds,
    ):
        self.now += seconds


@dataclass
class FakeTransactionResult:
    after: Observation


class FakeTransaction:
    def __init__(
        self,
        fail_at=None,
        *,
        clock=None,
        advance_each=0.0,
    ):
        self.calls = []
        self.results = []

        self.fail_at = fail_at
        self.clock = clock
        self.advance_each = (
            advance_each
        )

    def execute(
        self,
        action,
    ):
        self.calls.append(
            action
        )

        if self.clock is not None:
            self.clock.advance(
                self.advance_each
            )

        if (
            self.fail_at is not None
            and len(self.calls)
            == self.fail_at
        ):
            raise RuntimeError(
                "execution failed"
            )

        observation = Observation(
            screenshot=(
                f"screen-{len(self.calls)}"
            ),
            screen_width=2560,
            screen_height=1440,
        )

        result = (
            FakeTransactionResult(
                after=observation
            )
        )

        self.results.append(
            result
        )

        return result


def make_plan():
    return TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": (
                        "Type hello"
                    ),
                    "action": {
                        "action_type": (
                            "type_text"
                        ),
                        "text": "hello",
                    },
                },
                {
                    "step_id": 2,
                    "description": (
                        "Press Enter"
                    ),
                    "action": {
                        "action_type": (
                            "key_press"
                        ),
                        "keys": [
                            "enter"
                        ],
                    },
                },
            ]
        }
    )


def test_agent_loop_success():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(
        make_plan()
    )

    assert (
        result.success
        is True
    )

    assert (
        runtime.state.done
        is True
    )

    assert (
        runtime.state.failed
        is False
    )

    assert (
        runtime.state.step_count
        == 2
    )

    assert len(
        runtime.state.history
    ) == 2

    assert len(
        transaction.calls
    ) == 2

    assert (
        runtime.state
        .history[0]
        .success
        is True
    )

    assert (
        runtime.state
        .history[1]
        .success
        is True
    )


def test_agent_loop_updates_latest_observation():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(
        make_plan()
    )

    assert (
        result.success
        is True
    )

    assert (
        runtime.state.observation
        is transaction.results[-1].after
    )


def test_agent_loop_execution_failure():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction(
            fail_at=2
        )
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(
        make_plan()
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        runtime.state.done
        is False
    )

    assert (
        runtime.state.step_count
        == 2
    )

    assert (
        runtime.state
        .history[0]
        .success
        is True
    )

    assert (
        runtime.state
        .history[1]
        .success
        is False
    )

    assert (
        "execution failed"
        in result.failure_reason
    )


def test_agent_loop_max_steps():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=1,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(
        make_plan()
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        runtime.state.step_count
        == 1
    )

    assert len(
        transaction.calls
    ) == 1

    assert (
        "max_steps exceeded"
        in result.failure_reason
    )


def test_agent_loop_timeout_after_last_action(
    monkeypatch,
):
    clock = FakeClock(
        0.0
    )

    monkeypatch.setattr(
        runtime_module.time,
        "monotonic",
        clock.monotonic,
    )

    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
        timeout_s=1.0,
    )

    transaction = FakeTransaction(
        clock=clock,
        advance_each=0.6,
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(
        make_plan()
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        runtime.state.done
        is False
    )

    assert len(
        transaction.calls
    ) == 2

    assert (
        "timeout exceeded"
        in result.failure_reason
    )

    # The GUI action itself completed.
    # The overall task failed because
    # the total time budget was exceeded.
    assert (
        runtime.state
        .history[-1]
        .success
        is True
    )


def test_agent_loop_final_check_success():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    def final_check():
        return SuccessJudgeResult(
            success=True,
            reason="target found",
            matched_text="hello",
        )

    result = loop.run(
        make_plan(),
        final_check=final_check,
    )

    assert (
        result.success
        is True
    )

    assert (
        result.judge_result
        is not None
    )

    assert (
        result.judge_result.success
        is True
    )

    assert (
        runtime.state.done
        is True
    )


def test_agent_loop_final_check_failure():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    def final_check():
        return SuccessJudgeResult(
            success=False,
            reason="target missing",
        )

    result = loop.run(
        make_plan(),
        final_check=final_check,
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        runtime.state.done
        is False
    )

    assert (
        result.failure_reason
        == "target missing"
    )

    assert (
        result.judge_result
        is not None
    )

    assert (
        result.judge_result.success
        is False
    )


def test_agent_loop_timeout_during_final_check(
    monkeypatch,
):
    clock = FakeClock(
        0.0
    )

    monkeypatch.setattr(
        runtime_module.time,
        "monotonic",
        clock.monotonic,
    )

    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
        timeout_s=1.0,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    def final_check():
        clock.advance(
            1.5
        )

        return SuccessJudgeResult(
            success=True,
            reason="target found",
        )

    result = loop.run(
        make_plan(),
        final_check=final_check,
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        runtime.state.done
        is False
    )

    assert (
        "timeout exceeded"
        in result.failure_reason
    )

    assert (
        result.judge_result
        is not None
    )

    assert (
        result.judge_result.success
        is True
    )


def test_agent_loop_final_check_exception_uses_same_failure_reason():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    def final_check():
        raise RuntimeError(
            "OCR failed"
        )

    result = loop.run(
        make_plan(),
        final_check=final_check,
    )

    expected = (
        "success judge failed: "
        "OCR failed"
    )

    assert (
        result.success
        is False
    )

    assert (
        result.failure_reason
        == expected
    )

    assert (
        runtime.state.failure_reason
        == expected
    )

    assert (
        runtime.state.failed
        is True
    )


def test_agent_loop_rejects_invalid_final_check_result():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = (
        FakeTransaction()
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    def final_check():
        return None

    result = loop.run(
        make_plan(),
        final_check=final_check,
    )

    assert (
        result.success
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        "final_check must return "
        "SuccessJudgeResult"
        in result.failure_reason
    )

    assert (
        runtime.state.failure_reason
        == result.failure_reason
    )