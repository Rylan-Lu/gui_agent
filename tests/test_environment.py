import pytest

import gui_agent.agent.environment as environment_module
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
        return (
            self.region[2],
            self.region[3],
        )


class FakeScreenCapture:
    def __init__(
        self,
        events=None,
    ):
        self.last_region = None
        self.closed = False
        self.capture_calls = 0
        self.events = events

    def capture(
        self,
        region=None,
    ):
        self.capture_calls += 1
        self.last_region = region

        if self.events is not None:
            self.events.append(
                "capture"
            )

        if region is None:
            region = (
                0,
                0,
                2560,
                1440,
            )

        return FakeFrame(
            region=region
        )

    def close(self):
        self.closed = True


class FakeOCREngine:
    def __init__(
        self,
        events=None,
    ):
        self.calls = []
        self.events = events

    def recognize(
        self,
        image,
    ):
        self.calls.append(
            image
        )

        if self.events is not None:
            self.events.append(
                "ocr"
            )

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

    obs = env.observe(
        use_ocr=True
    )

    assert (
        capture.capture_calls
        == 1
    )

    assert len(
        ocr.calls
    ) == 1

    assert (
        obs.ocr_result[0]["text"]
        == "Firefox"
    )

    assert (
        obs.metadata["image_size"]
        == (
            2560,
            1440,
        )
    )


def test_environment_ocr_requires_engine_before_capture():
    capture = FakeScreenCapture()

    env = DesktopEnvironment(
        screen_capture=capture
    )

    with pytest.raises(
        RuntimeError,
        match="no OCR engine",
    ):
        env.observe(
            use_ocr=True
        )

    # Configuration failure must happen before screenshot capture.
    assert (
        capture.capture_calls
        == 0
    )


def test_environment_observe_fullscreen():
    capture = FakeScreenCapture()

    env = DesktopEnvironment(
        capture
    )

    obs = env.observe()

    assert (
        obs.screenshot
        == "fake-image"
    )

    assert (
        obs.screen_width
        == 2560
    )

    assert (
        obs.screen_height
        == 1440
    )

    assert (
        obs.metadata["region"]
        == (
            0,
            0,
            2560,
            1440,
        )
    )

    assert (
        obs.metadata["image_size"]
        == (
            2560,
            1440,
        )
    )


def test_environment_observe_region():
    capture = FakeScreenCapture()

    env = DesktopEnvironment(
        capture
    )

    obs = env.observe(
        (
            100,
            200,
            800,
            600,
        )
    )

    assert (
        capture.last_region
        == (
            100,
            200,
            800,
            600,
        )
    )

    assert (
        obs.screen_width
        == 800
    )

    assert (
        obs.screen_height
        == 600
    )

    assert (
        obs.metadata["region"]
        == (
            100,
            200,
            800,
            600,
        )
    )

    assert (
        obs.metadata["image_size"]
        == (
            800,
            600,
        )
    )


def test_observation_timestamp_is_recorded_before_ocr(
    monkeypatch,
):
    events = []

    capture = FakeScreenCapture(
        events=events
    )

    ocr = FakeOCREngine(
        events=events
    )

    def fake_monotonic():
        events.append(
            "timestamp"
        )

        return 123.456

    monkeypatch.setattr(
        environment_module.time,
        "monotonic",
        fake_monotonic,
    )

    env = DesktopEnvironment(
        screen_capture=capture,
        ocr_engine=ocr,
    )

    obs = env.observe(
        use_ocr=True
    )

    assert events == [
        "capture",
        "timestamp",
        "ocr",
    ]

    assert (
        obs.timestamp
        == 123.456
    )


def test_environment_without_ocr_does_not_call_engine():
    capture = FakeScreenCapture()
    ocr = FakeOCREngine()

    env = DesktopEnvironment(
        screen_capture=capture,
        ocr_engine=ocr,
    )

    obs = env.observe(
        use_ocr=False
    )

    assert (
        capture.capture_calls
        == 1
    )

    assert (
        ocr.calls
        == []
    )

    assert (
        obs.ocr_result
        is None
    )


def test_environment_reuses_same_ocr_engine():
    capture = FakeScreenCapture()
    ocr = FakeOCREngine()

    env = DesktopEnvironment(
        screen_capture=capture,
        ocr_engine=ocr,
    )

    env.observe(
        use_ocr=True
    )

    env.observe(
        use_ocr=True
    )

    assert (
        env.ocr_engine
        is ocr
    )

    assert len(
        ocr.calls
    ) == 2

    assert (
        capture.capture_calls
        == 2
    )


def test_environment_close():
    capture = FakeScreenCapture()

    env = DesktopEnvironment(
        capture
    )

    env.close()

    assert (
        capture.closed
        is True
    )


def test_environment_context_manager_closes_capture():
    capture = FakeScreenCapture()

    with DesktopEnvironment(
        capture
    ) as env:
        assert (
            env.screen_capture
            is capture
        )

        assert (
            capture.closed
            is False
        )

    assert (
        capture.closed
        is True
    )