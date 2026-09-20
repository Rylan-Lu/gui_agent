from pathlib import Path

import pytest

from gui_agent.models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
)


def test_model_request():
    request = ModelRequest(
        prompt="Describe the screen",
        image_path=Path("screen.png"),
    )

    assert request.prompt == "Describe the screen"
    assert request.image_path == Path("screen.png")


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        ModelRequest(prompt="   ")


def test_model_response():
    response = ModelResponse(
        text="A browser is open.",
        model_name="test-model",
    )

    assert response.text == "A browser is open."
    assert response.model_name == "test-model"


def test_empty_response_rejected():
    with pytest.raises(ValueError):
        ModelResponse(
            text="",
            model_name="test-model",
        )


def test_model_client_protocol():
    class FakeModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                text=f"Received: {request.prompt}",
                model_name="fake-model",
            )

    model: ModelClient = FakeModel()

    response = model.generate(
        ModelRequest(prompt="Hello")
    )

    assert response.text == "Received: Hello"
    assert response.model_name == "fake-model"