import base64
import io
import json
from urllib.error import HTTPError, URLError

import pytest
from PIL import Image

import gui_agent.models.api_model as api_module

from gui_agent.models.api_model import APIModelClient
from gui_agent.models.base import ModelRequest


ENDPOINT = "https://example.com/v1/chat/completions"
MODEL = "test-vision-model"


@pytest.fixture
def client():
    return APIModelClient(
        endpoint=ENDPOINT,
        model_id=MODEL,
        timeout=12.0,
    )


def mock_response(monkeypatch, response_data):
    """Replace the network request with an in-memory response."""

    def fake_urlopen(request, timeout):
        return io.BytesIO(
            json.dumps(response_data).encode("utf-8")
        )

    monkeypatch.setattr(
        api_module,
        "urlopen",
        fake_urlopen,
    )


def test_missing_api_key(client, monkeypatch):
    monkeypatch.delenv(
        "GUI_AGENT_API_KEY",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="not configured"):
        client.generate(
            ModelRequest(prompt="Hello")
        )


def test_multimodal_payload(client, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    image_path = tmp_path / "test.png"

    Image.new(
        "RGB",
        (32, 32),
        "white",
    ).save(image_path)

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout

        return io.BytesIO(
            json.dumps({
                "model": MODEL,
                "choices": [{
                    "message": {
                        "content": "TEST-42"
                    },
                    "finish_reason": "stop",
                }],
            }).encode("utf-8")
        )

    monkeypatch.setattr(
        api_module,
        "urlopen",
        fake_urlopen,
    )

    response = client.generate(
        ModelRequest(
            prompt="Read the image",
            image_path=image_path,
            system_prompt="You are an OCR assistant.",
        )
    )

    payload = json.loads(
        captured["request"].data
    )

    assert captured["timeout"] == 12.0
    assert payload["model"] == MODEL

    assert payload["messages"][0]["role"] == "system"

    content = payload["messages"][1]["content"]

    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Read the image"

    assert content[1]["type"] == "image_url"

    image_url = content[1]["image_url"]["url"]

    assert image_url.startswith(
        "data:image/png;base64,"
    )

    encoded = image_url.split(",", 1)[1]

    decoded = base64.b64decode(encoded)

    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    assert response.text == "TEST-42"


def test_text_only_request(client, monkeypatch):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)

        return io.BytesIO(
            json.dumps({
                "choices": [{
                    "message": {
                        "content": "Hello"
                    }
                }]
            }).encode("utf-8")
        )

    monkeypatch.setattr(
        api_module,
        "urlopen",
        fake_urlopen,
    )

    response = client.generate(
        ModelRequest(prompt="Hello")
    )

    content = captured["payload"]["messages"][0]["content"]

    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert response.text == "Hello"
    assert response.model_name == MODEL


def test_missing_image(client, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        client.generate(
            ModelRequest(
                prompt="Describe this image",
                image_path=missing,
            )
        )


@pytest.mark.parametrize(
    "response_data",
    [
        {"choices": []},
        {"choices": [{}]},
    ],
)
def test_invalid_response_structure(
    client,
    monkeypatch,
    response_data,
):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    mock_response(monkeypatch, response_data)

    with pytest.raises(
        RuntimeError,
        match="Invalid API response structure",
    ):
        client.generate(
            ModelRequest(prompt="Hello")
        )


def test_truncated_response(client, monkeypatch):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    mock_response(
        monkeypatch,
        {
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": "Incomplete"
                },
            }]
        },
    )

    with pytest.raises(
        RuntimeError,
        match="truncated",
    ):
        client.generate(
            ModelRequest(prompt="Hello")
        )


def test_http_error(client, monkeypatch):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=ENDPOINT,
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b""),
        )

    monkeypatch.setattr(
        api_module,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(HTTPError) as exc:
        client.generate(
            ModelRequest(prompt="Hello")
        )

    assert exc.value.code == 429


def test_network_error(client, monkeypatch):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    def fake_urlopen(request, timeout):
        raise URLError("Connection unavailable")

    monkeypatch.setattr(
        api_module,
        "urlopen",
        fake_urlopen,
    )

    with pytest.raises(URLError):
        client.generate(
            ModelRequest(prompt="Hello")
        )


def test_invalid_response_content(client, monkeypatch):
    monkeypatch.setenv(
        "GUI_AGENT_API_KEY",
        "test-key",
    )

    mock_response(
        monkeypatch,
        {
            "choices": [{
                "message": {
                    "content": None
                }
            }]
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid API response content",
    ):
        client.generate(
            ModelRequest(prompt="Hello")
        )