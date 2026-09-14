from __future__ import annotations

import numpy as np

from gui_agent.ocr.base import BoundingBox, OCREngine, OCRResult
from gui_agent.runtime.gpu_runtime import initialize_torch_gpu_runtime


class PaddleOCREngine(OCREngine):
    def __init__(
        self,
        *,
        lang: str = "ch",
        device: str = "gpu:0",
        min_confidence: float = 0.0,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self.min_confidence = min_confidence

        if device.startswith("gpu"):
            initialize_torch_gpu_runtime()

        # Lazy import is intentional. On the validated Windows setup, Torch
        # must initialize before Paddle/PaddleOCR is loaded.
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        if image is None:
            raise ValueError("image is None")
        if image.ndim != 3:
            raise ValueError("image must be a 3D array")

        predictions = self._ocr.predict(image)
        results: list[OCRResult] = []

        for prediction in predictions:
            data = prediction.json["res"]
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])

            for text, score, box in zip(texts, scores, boxes):
                text = str(text).strip()
                confidence = float(score)

                if not text or confidence < self.min_confidence:
                    continue

                box_array = np.asarray(box, dtype=float).reshape(-1)
                if box_array.size != 4:
                    continue

                x1, y1, x2, y2 = box_array
                bbox: BoundingBox = (
                    (float(x1), float(y1)),
                    (float(x2), float(y1)),
                    (float(x2), float(y2)),
                    (float(x1), float(y2)),
                )

                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

        return results
