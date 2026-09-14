from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from mss import MSS

Region: TypeAlias = tuple[int, int, int, int]
ImageSize: TypeAlias = tuple[int, int]


@dataclass(slots=True)
class CapturedFrame:
    image: np.ndarray
    region: Region

    @property
    def image_size(self) -> ImageSize:
        height, width = self.image.shape[:2]
        return width, height


class ScreenCapture:
    def __init__(self) -> None:
        self._backend = MSS()

    def capture(self, region: Region | None = None) -> CapturedFrame:
        if region is None:
            monitor = self._backend.primary_monitor
            actual_region: Region = (
                monitor["left"],
                monitor["top"],
                monitor["width"],
                monitor["height"],
            )
        else:
            self._validate_region(region)
            left, top, width, height = region
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
            actual_region = region

        screenshot = self._backend.grab(monitor)
        image = np.asarray(screenshot)[:, :, :3].copy()
        return CapturedFrame(image=image, region=actual_region)

    def _validate_region(self, region: Region) -> None:
        left, top, width, height = region

        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than zero")

        monitor = self._backend.primary_monitor
        screen_left = monitor["left"]
        screen_top = monitor["top"]
        screen_right = screen_left + monitor["width"]
        screen_bottom = screen_top + monitor["height"]

        region_right = left + width
        region_bottom = top + height

        if (
            left < screen_left
            or top < screen_top
            or region_right > screen_right
            or region_bottom > screen_bottom
        ):
            raise ValueError("Region must be within the primary screen")

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
