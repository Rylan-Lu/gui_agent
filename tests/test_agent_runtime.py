import math

import pytest

import gui_agent.agent.runtime as runtime_module
from gui_agent.agent.observation import Observation
from gui_agent.agent.runtime import (
    AgentRuntime,
    AgentState,
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


def test_create_runtime():
    runtime = AgentRuntime.create(
        task="Open Firefox",
        max_steps=5,
        timeout_s=30.0,
    )

    assert (
        runtime.state.task
        == "Open Firefox"
    )

    assert (
        runtime.state.max_steps
        == 5
    )

    assert (
        runtime.state.timeout_s
        == 30.0
    )

    assert (
        runtime.state.step_count
        == 0
    )

    assert (
        runtime.state.current_step
        == 0
    )

    assert (
        runtime.state.history
        == []
    )

    assert (
        runtime.state.done
        is False
    )

    assert (
        runtime.state.failed
        is False
    )


def test_start_runtime(
    monkeypatch,
):
    clock = FakeClock(
        10.0
    )

    monkeypatch.setattr(
        runtime_module.time,
        "monotonic",
        clock.monotonic,
    )

    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.start()

    assert (
        runtime.state.started_at
        == 10.0
    )

    clock.advance(
        2.5
    )

    assert (
        runtime.elapsed_s
        == pytest.approx(
            2.5
        )
    )


def test_start_is_idempotent(
    monkeypatch,
):
    clock = FakeClock(
        10.0
    )

    monkeypatch.setattr(
        runtime_module.time,
        "monotonic",
        clock.monotonic,
    )

    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.start()

    clock.advance(
        5.0
    )

    runtime.start()

    assert (
        runtime.state.started_at
        == 10.0
    )


def test_update_observation():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    observation = Observation(
        screenshot="screenshot.png",
        screen_width=2560,
        screen_height=1440,
    )

    runtime.update_observation(
        observation
    )

    assert (
        runtime.state.observation
        is observation
    )


def test_begin_and_finish_step():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.start()

    step = runtime.begin_step(
        "Launch Firefox"
    )

    assert (
        runtime.state.step_count
        == 1
    )

    assert (
        runtime.state.current_step
        == 1
    )

    assert len(
        runtime.state.history
    ) == 1

    assert (
        step.description
        == "Launch Firefox"
    )

    assert (
        step.success
        is None
    )

    runtime.finish_step(
        success=True
    )

    assert (
        step.success
        is True
    )

    assert (
        step.error
        is None
    )


def test_begin_step_rejects_active_previous_step():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.begin_step(
        "First step"
    )

    with pytest.raises(
        RuntimeError,
        match="still active",
    ):
        runtime.begin_step(
            "Second step"
        )

    assert (
        runtime.state.step_count
        == 1
    )

    assert len(
        runtime.state.history
    ) == 1

    assert (
        runtime.state
        .history[0]
        .success
        is None
    )


def test_finish_step_requires_active_step():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="no active step",
    ):
        runtime.finish_step(
            success=True
        )


def test_completed_step_cannot_be_finished_twice():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.begin_step(
        "First step"
    )

    runtime.finish_step(
        success=True
    )

    with pytest.raises(
        RuntimeError,
        match="no active step",
    ):
        runtime.finish_step(
            success=False
        )

    assert (
        runtime.state
        .history[-1]
        .success
        is True
    )


def test_max_steps_limit():
    runtime = AgentRuntime.create(
        "Open Firefox",
        max_steps=1,
    )

    runtime.start()

    runtime.begin_step(
        "First step"
    )

    runtime.finish_step(
        success=True
    )

    assert (
        runtime.check_limits()
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
        "max_steps exceeded"
        in runtime.state.failure_reason
    )


def test_check_timeout_does_not_apply_max_steps(
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
        "Open Firefox",
        max_steps=1,
        timeout_s=10.0,
    )

    runtime.start()

    runtime.begin_step(
        "Only step"
    )

    runtime.finish_step(
        success=True
    )

    assert (
        runtime.state.step_count
        == 1
    )

    assert (
        runtime.check_timeout()
        is True
    )

    assert (
        runtime.state.failed
        is False
    )


def test_timeout_limit(
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
        "Open Firefox",
        timeout_s=1.0,
    )

    runtime.start()

    clock.advance(
        2.0
    )

    assert (
        runtime.check_timeout()
        is False
    )

    assert (
        runtime.state.failed
        is True
    )

    assert (
        "timeout exceeded"
        in runtime.state.failure_reason
    )


def test_mark_done():
    runtime = (
        AgentRuntime.create(
            "Open Firefox"
        )
    )

    runtime.start()
    runtime.mark_done()

    assert (
        runtime.state.done
        is True
    )

    assert (
        runtime.state.failed
        is False
    )

    assert (
        runtime.check_limits()
        is False
    )


@pytest.mark.parametrize(
    "task",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_task(
    task,
):
    with pytest.raises(
        ValueError
    ):
        AgentState(
            task=task
        )


@pytest.mark.parametrize(
    "max_steps",
    [
        0,
        -1,
        True,
        1.5,
        "5",
        None,
    ],
)
def test_invalid_max_steps(
    max_steps,
):
    with pytest.raises(
        ValueError
    ):
        AgentState(
            task="test",
            max_steps=max_steps,
        )


@pytest.mark.parametrize(
    "timeout_s",
    [
        0,
        -1,
        True,
        "60",
        None,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_timeout(
    timeout_s,
):
    with pytest.raises(
        ValueError
    ):
        AgentState(
            task="test",
            timeout_s=timeout_s,
        )


def test_integer_timeout_is_normalized_to_float():
    state = AgentState(
        task="test",
        timeout_s=30,
    )

    assert (
        state.timeout_s
        == 30.0
    )

    assert isinstance(
        state.timeout_s,
        float,
    )


@pytest.mark.parametrize(
    "description",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_step_description(
    description,
):
    runtime = (
        AgentRuntime.create(
            "test"
        )
    )

    with pytest.raises(
        ValueError
    ):
        runtime.begin_step(
            description
        )