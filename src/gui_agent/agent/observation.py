from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(slots=True)
class Observation:
    screenshot: Any
    timestamp: float = field(default_factory=time.monotonic)

    screen_width: int | None = None
    screen_height: int | None = None

    ocr_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)