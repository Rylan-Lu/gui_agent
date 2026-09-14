import numpy as np

from gui_agent.ocr.base import BoundingBox, OCREngine, OCRResult


class EasyOCREngine(OCREngine):
    def __init__(self) -> None:
        # Lazy import keeps EasyOCR/Torch out of the process unless this
        # fallback engine is actually selected.
        import easyocr

        self._reader = easyocr.Reader(["ch_sim", "en"], gpu=True)

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
