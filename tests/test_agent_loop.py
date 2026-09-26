from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime


class FakeTransaction:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def execute(self, action):
        self.calls.append(action)

        if (
            self.fail_at is not None
            and len(self.calls) == self.fail_at
        ):
            raise RuntimeError("execution failed")

        return f"result-{len(self.calls)}"


def make_plan():
    return TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Type hello",
                    "action": {
                        "action_type": "type_text",
                        "text": "hello",
                    },
                },
                {
                    "step_id": 2,
                    "description": "Press Enter",
                    "action": {
                        "action_type": "key_press",
                        "keys": ["enter"],
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

    transaction = FakeTransaction()

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(make_plan())

    assert result.success is True

    assert runtime.state.done is True
    assert runtime.state.failed is False

    assert runtime.state.step_count == 2
    assert len(runtime.state.history) == 2
    assert len(transaction.calls) == 2

    assert runtime.state.history[0].success is True
    assert runtime.state.history[1].success is True


def test_agent_loop_execution_failure():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = FakeTransaction(
        fail_at=2,
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(make_plan())

    assert result.success is False

    assert runtime.state.failed is True
    assert runtime.state.done is False

    assert runtime.state.step_count == 2

    assert runtime.state.history[0].success is True
    assert runtime.state.history[1].success is False

    assert "execution failed" in result.failure_reason


def test_agent_loop_max_steps():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=1,
    )

    transaction = FakeTransaction()

    loop = AgentLoop(
        runtime,
        transaction,
    )

    result = loop.run(make_plan())

    assert result.success is False
    assert runtime.state.failed is True

    assert runtime.state.step_count == 1
    assert len(transaction.calls) == 1

from gui_agent.agent.success_judge import SuccessJudgeResult


def test_agent_loop_final_check_success():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = FakeTransaction()

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

    assert result.success is True
    assert result.judge_result is not None
    assert result.judge_result.success is True
    assert runtime.state.done is True


def test_agent_loop_final_check_failure():
    runtime = AgentRuntime.create(
        "test task",
        max_steps=5,
    )

    transaction = FakeTransaction()

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

    assert result.success is False
    assert runtime.state.failed is True
    assert runtime.state.done is False
    assert result.failure_reason == "target missing"