import time

import pytest

from gui_agent.agent.runtime import AgentRuntime, AgentState


def test_create_runtime():
    runtime = AgentRuntime.create(
        task="Open Firefox",
        max_steps=5,
        timeout_s=30.0,
    )

    assert runtime.state.task == "Open Firefox"
    assert runtime.state.max_steps == 5
    assert runtime.state.timeout_s == 30.0

    assert runtime.state.step_count == 0
    assert runtime.state.current_step == 0

    assert runtime.state.history == []

    assert runtime.state.done is False
    assert runtime.state.failed is False


def test_start_runtime():
    runtime = AgentRuntime.create("Open Firefox")

    runtime.start()

    assert runtime.state.started_at is not None
    assert runtime.elapsed_s >= 0.0


from gui_agent.agent.observation import Observation


def test_update_observation():
    runtime = AgentRuntime.create("Open Firefox")

    observation = Observation(
        screenshot="screenshot.png",
        screen_width=2560,
        screen_height=1440,
    )

    runtime.update_observation(observation)

    assert runtime.state.observation is observation


def test_begin_and_finish_step():
    runtime = AgentRuntime.create("Open Firefox")

    runtime.start()

    step = runtime.begin_step("Launch Firefox")

    assert runtime.state.step_count == 1
    assert runtime.state.current_step == 1
    assert len(runtime.state.history) == 1

    assert step.description == "Launch Firefox"
    assert step.success is None

    runtime.finish_step(success=True)

    assert step.success is True
    assert step.error is None


def test_max_steps_limit():
    runtime = AgentRuntime.create(
        "Open Firefox",
        max_steps=1,
    )

    runtime.start()

    runtime.begin_step("First step")
    runtime.finish_step(success=True)

    assert runtime.check_limits() is False

    assert runtime.state.failed is True
    assert runtime.state.done is False

    assert "max_steps exceeded" in runtime.state.failure_reason


def test_timeout_limit():
    runtime = AgentRuntime.create(
        "Open Firefox",
        timeout_s=1.0,
    )

    runtime.start()

    runtime.state.started_at = time.monotonic() - 2.0

    assert runtime.check_limits() is False

    assert runtime.state.failed is True
    assert "timeout exceeded" in runtime.state.failure_reason


def test_mark_done():
    runtime = AgentRuntime.create("Open Firefox")

    runtime.start()
    runtime.mark_done()

    assert runtime.state.done is True
    assert runtime.state.failed is False
    assert runtime.check_limits() is False


def test_invalid_state():
    with pytest.raises(ValueError):
        AgentState(task="")

    with pytest.raises(ValueError):
        AgentState(
            task="test",
            max_steps=0,
        )

    with pytest.raises(ValueError):
        AgentState(
            task="test",
            timeout_s=0,
        )