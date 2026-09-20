import io
import json

import pytest

from unittest.mock import patch
from PIL import Image

from gui_agent.models.api_model import APIModelClient
from gui_agent.models.base import ModelRequest


ENDPOINT = "https://example.com/v1/chat/completions"


def make_client():
    return APIModelClient(
        endpoint=ENDPOINT,
        model_id="test-model",
    )


def test_invalid_endpoint():
    with pytest.raises(ValueError):
        APIModelClient(
            endpoint="http://example.com/v1/chat/completions",
            model_id="test-model",
        )


def test_text_messages():
    client = make_client()

    request = ModelRequest(
        prompt="Hello",
        system_prompt="You are an assistant.",
    )

    messages = client._build_messages(request)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][0]["text"] == "Hello"


def test_image_messages(tmp_path):
    image_path = tmp_path / "screen.jpg"

    Image.new("RGB", (32, 32)).save(image_path)

    request = ModelRequest(
        prompt="Describe this image",
        image_path=image_path,
    )

    messages = make_client()._build_messages(request)

    image_url = messages[0]["content"][1]["image_url"]["url"]

    assert image_url.startswith("data:image/png;base64,")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GUI_AGENT_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        make_client().generate(
            ModelRequest(prompt="Hello")
        )


def test_mock_api_response(monkeypatch):
    monkeypatch.setenv("GUI_AGENT_API_KEY", "fake-test-key")

    fake_response = {
        "model": "test-model",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": '{"steps":[{"step_id":1,"description":"Open browser"}]}'
            },
        }],
    }

    response_bytes = json.dumps(
        fake_response
    ).encode("utf-8")

    with patch(
        "gui_agent.models.api_model.urlopen",
        return_value=io.BytesIO(response_bytes),
    ) as mock_urlopen:

        result = make_client().generate(
            ModelRequest(prompt="Open browser")
        )

    assert result.model_name == "test-model"
    assert "Open browser" in result.text
    assert mock_urlopen.call_count == 1