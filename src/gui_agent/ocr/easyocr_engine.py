from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gui_agent.ocr.base import BoundingBox, OCREngine, OCRResult


class EasyOCREngine(OCREngine):
    """EasyOCR adapter used as a fallback/debug OCR backend."""

    def __init__(
        self,
        *,
        languages: Sequence[str] = ("ch_sim", "en"),
        gpu: bool = True,
    ) -> None:
        if not languages:
            raise ValueError("languages must not be empty")

        # Lazy import keeps EasyOCR/Torch out of the process unless this
        # fallback engine is actually selected.
        import easyocr

        self._reader = easyocr.Reader(list(languages), gpu=gpu)

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        raw_results = self._reader.readtext(image)

        results: list[OCRResult] = []
        for bbox, text, confidence in raw_results:
            converted_bbox: BoundingBox = tuple(
                (float(x), float(y)) for x, y in bbox
            )
            results.append(
                OCRResult(
                    text=text,
                    confidence=float(confidence),
                    bbox=converted_bbox,
                )
            )

        return results
