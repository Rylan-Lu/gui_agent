from gui_agent.agent.environment import DesktopEnvironment


class FakeFrame:
    def __init__(
        self,
        image="fake-image",
        region=(0, 0, 2560, 1440),
    ):
        self.image = image
        self.region = region

    @property
    def image_size(self):
        return self.region[2], self.region[3]


class FakeScreenCapture:
    def __init__(self):
        self.last_region = None
        self.closed = False

    def capture(self, region=None):
        self.last_region = region

        if region is None:
            region = (0, 0, 2560, 1440)

        return FakeFrame(region=region)

    def close(self):
        self.closed = True

class FakeOCREngine:
    def __init__(self):
        self.calls = []

    def recognize(self, image):
        self.calls.append(image)

        return [
            {
                "text": "Firefox",
                "confidence": 0.98,
            }
        ]


def test_environment_observe_with_ocr():
    capture = FakeScreenCapture()
    ocr = FakeOCREngine()

    env = DesktopEnvironment(
        screen_capture=capture,
        ocr_engine=ocr,
    )

    obs = env.observe(use_ocr=True)

    assert len(ocr.calls) == 1
    assert obs.ocr_result[0]["text"] == "Firefox"
    assert obs.metadata["image_size"] == (2560, 1440)


def test_environment_ocr_requires_engine():
    capture = FakeScreenCapture()
    env = DesktopEnvironment(screen_capture=capture)

    import pytest

    with pytest.raises(RuntimeError):
        env.observe(use_ocr=True)


def test_environment_observe_fullscreen():
    capture = FakeScreenCapture()
    env = DesktopEnvironment(capture)

    obs = env.observe()

    assert obs.screenshot == "fake-image"
    assert obs.screen_width == 2560
    assert obs.screen_height == 1440
    assert obs.metadata["region"] == (0, 0, 2560, 1440)


def test_environment_observe_region():
    capture = FakeScreenCapture()
    env = DesktopEnvironment(capture)

    obs = env.observe((100, 200, 800, 600))

    assert capture.last_region == (100, 200, 800, 600)
    assert obs.screen_width == 800
    assert obs.screen_height == 600
    assert obs.metadata["region"] == (100, 200, 800, 600)


def test_environment_close():
    capture = FakeScreenCapture()
    env = DesktopEnvironment(capture)

    env.close()

    assert capture.closed is True