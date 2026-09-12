from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Point: TypeAlias = tuple[float, float]

BoundingBox: TypeAlias = tuple[Point, Point, Point, Point]

@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    bbox: BoundingBox

class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Recognize text from a BGR image."""
        raise NotImplementedError
