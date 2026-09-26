from __future__ import annotations

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
        frame = self.screen_capture.capture(region)

        width, height = frame.image_size

        ocr_result = None

        if use_ocr:
            if self.ocr_engine is None:
                raise RuntimeError(
                    "OCR requested but no OCR engine is configured"
                )

            ocr_result = self.ocr_engine.recognize(frame.image)

        return Observation(
            screenshot=frame.image,
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

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()