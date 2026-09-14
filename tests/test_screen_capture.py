import sys
import types

import numpy as np
import pytest

# Keep unit tests independent of the real desktop/MSS runtime.
fake_mss = types.ModuleType("mss")
fake_mss.MSS = object
sys.modules.setdefault("mss", fake_mss)

from gui_agent.perception.screen_capture import CapturedFrame, ScreenCapture


class FakeBackend:
    def __init__(self, monitor=None):
        self.primary_monitor = monitor or {"left": 0, "top": 0, "width": 100, "height": 80}
        self.grab_calls = []
        self.closed = False

    def grab(self, monitor):
        self.grab_calls.append(dict(monitor))
        h = monitor["height"]
        w = monitor["width"]
        image = np.zeros((h, w, 4), dtype=np.uint8)
        image[..., 0] = 1
        image[..., 1] = 2
        image[..., 2] = 3
        image[..., 3] = 255
        return image

    def close(self):
        self.closed = True


def make_capture(monitor=None):
    capture = object.__new__(ScreenCapture)
    capture._backend = FakeBackend(monitor)
    return capture


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 3), (1, 1)),
        ((10, 20, 3), (20, 10)),
        ((720, 1280, 3), (1280, 720)),
        ((1440, 2560, 3), (2560, 1440)),
    ],
)
def test_captured_frame_image_size(shape, expected):
    frame = CapturedFrame(np.zeros(shape, dtype=np.uint8), (0, 0, expected[0], expected[1]))
    assert frame.image_size == expected


def test_capture_full_primary_monitor():
    capture = make_capture({"left": 10, "top": 20, "width": 40, "height": 30})
    frame = capture.capture()
    assert frame.region == (10, 20, 40, 30)
    assert frame.image.shape == (30, 40, 3)
    assert frame.image[0, 0].tolist() == [1, 2, 3]
    assert capture._backend.grab_calls == [{"left": 10, "top": 20, "width": 40, "height": 30}]


@pytest.mark.parametrize(
    "region",
    [
        (0, 0, 1, 1),
        (0, 0, 100, 80),
        (10, 10, 20, 20),
        (99, 79, 1, 1),
        (50, 40, 50, 40),
    ],
)
def test_capture_valid_regions(region):
    capture = make_capture()
    frame = capture.capture(region)
    assert frame.region == region
    assert frame.image_size == (region[2], region[3])


@pytest.mark.parametrize(
    "region",
    [
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, -1, 1),
        (0, 0, 1, -1),
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (90, 0, 11, 10),
        (0, 70, 10, 11),
        (101, 0, 1, 1),
        (0, 81, 1, 1),
    ],
)
def test_invalid_regions(region):
    capture = make_capture()
    with pytest.raises(ValueError):
        capture.capture(region)


def test_region_validation_supports_negative_monitor_origin():
    capture = make_capture({"left": -100, "top": -50, "width": 100, "height": 80})
    frame = capture.capture((-100, -50, 100, 80))
    assert frame.region == (-100, -50, 100, 80)


def test_close_closes_backend():
    capture = make_capture()
    capture.close()
    assert capture._backend.closed is True


def test_context_manager_exit_closes_backend():
    capture = make_capture()
    with capture as entered:
        assert entered is capture
    assert capture._backend.closed is True
