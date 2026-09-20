from pathlib import Path

import pytest
from PIL import Image

from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import LocalVLMClient


def test_lazy_loading():
    """Creating a client must not load the model."""

    client = LocalVLMClient()

    assert client.model is None
    assert client.processor is None


def test_invalid_max_new_tokens():
    with pytest.raises(ValueError):
        LocalVLMClient(max_new_tokens=0)


def test_text_only_request():
    request = ModelRequest(
        prompt="Hello"
    )

    messages = LocalVLMClient._build_messages(
        request
    )

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][-1]["text"] == "Hello"


def test_missing_image_rejected(tmp_path):
    request = ModelRequest(
        prompt="Describe this image",
        image_path=tmp_path / "missing.png",
    )

    with pytest.raises(FileNotFoundError):
        LocalVLMClient._build_messages(request)


def test_valid_image_request(tmp_path):
    """A valid image must be converted into an RGB image."""

    image_path = tmp_path / "test.png"

    Image.new(
        "RGB",
        (100, 100),
    ).save(image_path)

    request = ModelRequest(
        prompt="Describe the screen",
        image_path=image_path,
    )

    messages = LocalVLMClient._build_messages(
        request
    )

    content = messages[-1]["content"]

    assert content[0]["type"] == "image"
    assert isinstance(content[0]["image"], Image.Image)
    assert content[0]["image"].mode == "RGB"

    assert content[-1]["text"] == "Describe the screen"