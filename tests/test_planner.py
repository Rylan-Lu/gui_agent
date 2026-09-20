from pathlib import Path

import pytest

from gui_agent.agent.planner import (
    Planner,
    PlannerOutputError,
)
from gui_agent.models.base import (
    ModelRequest,
    ModelResponse,
)


VALID_JSON = """
{
    "steps": [
        {
            "step_id": 1,
            "description": "Open browser"
        },
        {
            "step_id": 2,
            "description": "Search GUI Agent"
        }
    ]
}
"""


class FakeModel:
    """Model stub that records the received request."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_request = None
        self.call_count = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        self.call_count += 1

        return ModelResponse(
            text=self.response_text,
            model_name="fake-model",
        )


def test_planner_generates_valid_plan():
    model = FakeModel(VALID_JSON)
    planner = Planner(model)

    plan = planner.plan("Open browser and search GUI Agent")

    assert len(plan.steps) == 2
    assert plan.steps[0].step_id == 1
    assert plan.steps[1].description == "Search GUI Agent"


def test_planner_passes_instruction():
    model = FakeModel(VALID_JSON)

    Planner(model).plan("Open Firefox")

    assert "Open Firefox" in model.last_request.prompt
    assert model.last_request.system_prompt is not None


def test_planner_passes_screenshot():
    model = FakeModel(VALID_JSON)
    image = Path("screen.png")

    Planner(model).plan(
        instruction="Describe the screen",
        screenshot_path=image,
    )

    assert model.last_request.image_path == image


def test_planner_supports_text_only():
    model = FakeModel(VALID_JSON)

    Planner(model).plan("Open browser")

    assert model.last_request.image_path is None


def test_planner_accepts_json_fence():
    model = FakeModel(
        "```json\n" + VALID_JSON + "\n```"
    )

    plan = Planner(model).plan("Open browser")

    assert len(plan.steps) == 2


def test_planner_rejects_invalid_json():
    model = FakeModel("This is not JSON")

    with pytest.raises(PlannerOutputError):
        Planner(model).plan("Open browser")


def test_planner_rejects_invalid_plan():
    model = FakeModel(
        '{"steps": [{"step_id": 3, '
        '"description": "Open browser"}]}'
    )

    with pytest.raises(PlannerOutputError):
        Planner(model).plan("Open browser")


def test_planner_rejects_empty_instruction():
    model = FakeModel(VALID_JSON)

    with pytest.raises(ValueError):
        Planner(model).plan("   ")

    assert model.call_count == 0


def test_planner_calls_model_once():
    model = FakeModel(VALID_JSON)

    Planner(model).plan("Open browser")

    assert model.call_count == 1


def test_planner_rejects_extra_text():
    model = FakeModel(
        "Here is your plan:\n" + VALID_JSON
    )

    with pytest.raises(PlannerOutputError):
        Planner(model).plan("Open browser")