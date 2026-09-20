import pytest

from gui_agent.agent.chain import build_planning_chain
from gui_agent.agent.planner import Planner
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.models.base import ModelResponse


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1

        return ModelResponse(
            text='{"steps":[{"step_id":1,"description":"Open browser"}]}',
            model_name="fake-model",
        )


def test_agent_chain():
    model = FakeModel()
    planner = Planner(model)
    chain = build_planning_chain(planner)

    result = chain.invoke({
        "instruction": "Open browser",
    })

    assert isinstance(result, TaskPlan)
    assert len(result.steps) == 1
    assert model.calls == 1


def test_invalid_input():
    model = FakeModel()
    chain = build_planning_chain(Planner(model))

    with pytest.raises(ValueError):
        chain.invoke({
            "instruction": "",
        })

    assert model.calls == 0