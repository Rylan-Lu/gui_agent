from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gui_agent.ocr.base import OCRResult


@dataclass(slots=True)
class Observation:
    """Snapshot of the desktop state observed by the agent."""

    screenshot: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)

    screen_width: int | None = None
    screen_height: int | None = None

    ocr_result: list[OCRResult] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
