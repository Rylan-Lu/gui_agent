from __future__ import annotations

import time
from typing import Any

from gui_agent.agent.observation import Observation
from gui_agent.perception.screen_capture import ScreenCapture


class DesktopEnvironment:
    def __init__(
        self,
        screen_capture: Any | None = None,
        ocr_engine: Any | None = None,
    ):
        self.screen_capture = screen_capture or ScreenCapture()
        self.ocr_engine = ocr_engine

    def observe(
        self,
        region: tuple[int, int, int, int] | None = None,
        *,
        use_ocr: bool = False,
    ) -> Observation:
        """
        Capture the current desktop state.

        Observation.timestamp represents the age of the captured
        visual state, not the time at which optional OCR finishes.
        """

        # Configuration errors should fail before any screenshot work.
        if use_ocr and self.ocr_engine is None:
            raise RuntimeError(
                "OCR requested but no OCR engine is configured"
            )

        frame = self.screen_capture.capture(region)

        # Record the timestamp immediately after the screenshot is
        # obtained and before potentially expensive OCR inference.
        captured_at = time.monotonic()

        width, height = frame.image_size

        ocr_result = None

        if use_ocr:
            ocr_result = self.ocr_engine.recognize(
                frame.image
            )

        return Observation(
            screenshot=frame.image,
            timestamp=captured_at,
            screen_width=width,
            screen_height=height,
            ocr_result=ocr_result,
            metadata={
                "region": frame.region,
                "image_size": frame.image_size,
            },
        )

    def close(self) -> None:
        self.screen_capture.close()

    def __enter__(self) -> "DesktopEnvironment":
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        self.close()