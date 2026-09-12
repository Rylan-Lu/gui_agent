import easyocr
import numpy as np

from gui_agent.ocr.base import OCREngine, BoundingBox, OCRResult

class EasyOCR_ENGINE(OCREngine):
    def __init__(self) -> None:
        self._reader = easyocr.Reader(["ch_sim", "en"], gpu=True)

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        raw_results = self._reader.readtext(image)

        results: list[OCRResult] = []

        for bbox, text, confidence in raw_results:
            converted_bbox: BoundingBox = tuple((float(x), float(y)) for x, y in bbox)

            results.append(OCRResult(text=text, confidence=confidence, bbox=converted_bbox))

        return results
