"""Small VLM screenshot resource tests.  Never touch desktop or GPU."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gui_agent.agent.replan import temporary_replan_screenshot


def _obs(width: int, height: int):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    return SimpleNamespace(screenshot=pixels)


def test_large_replan_capture_is_downscaled():
    with temporary_replan_screenshot(_obs(1920, 1080)) as path:
        assert path.exists()
        with Image.open(path) as im:
            assert im.size == (1280, 720)
    assert not path.exists()


def test_small_replan_capture_keeps_original_size():
    with temporary_replan_screenshot(_obs(320, 160)) as path:
        with Image.open(path) as im:
            assert im.size == (320, 160)


def test_small_custom_max_edge_is_supported():
    with temporary_replan_screenshot(_obs(1600, 900), max_edge=800) as path:
        with Image.open(path) as im:
            assert im.size == (800, 450)


@pytest.mark.parametrize("edge", [True, False, 0, -1, 1280.0, "1280"])
def test_invalid_max_edge_is_rejected(edge):
    with pytest.raises(ValueError, match="max_edge"):
        with temporary_replan_screenshot(_obs(30, 20), max_edge=edge):
            pytest.fail("should not yield")


def test_temporary_screenshot_deleted_when_consumer_throws():
    saved_path: Path | None = None
    with pytest.raises(RuntimeError, match="test exception"):
        with temporary_replan_screenshot(_obs(1400, 800)) as path:
            saved_path = path
            assert path.exists()
            raise RuntimeError("test exception")
    assert saved_path is not None and not saved_path.exists()


def test_bgr_to_rgb_is_preserved():
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[0, 0] = [11, 22, 33]  # B, G, R
    with temporary_replan_screenshot(SimpleNamespace(screenshot=pixels)) as path:
        with Image.open(path) as im:
            assert im.getpixel((0, 0)) == (33, 22, 11)
