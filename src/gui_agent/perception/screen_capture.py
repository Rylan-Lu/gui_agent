from typing import TypeAlias

from mss import MSS
import numpy as np

Region: TypeAlias = tuple[int, int, int, int]

class ScreenCapture:
    def __init__(self) -> None:
        self._backend = MSS()

    def capture(self, region: Region | None=None) -> np.ndarray:
        if region is None:
            monitor = self._backend.primary_monitor
        else:
            left, top, width, height = region

            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be greater than zero")

            monitor = {"left": left, "top": top, "width": width, "height": height}

        screenshot = self._backend.grab(monitor)

        raw = np.asarray(screenshot)
        image = raw[:, :, :3].copy()

        return image

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()