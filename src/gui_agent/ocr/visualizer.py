from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gui_agent.ocr.base import OCRResult


class OCRVisualizer:
    def __init__(
        self,
        min_confidence: float = 0.3,
        font_size: int = 18,
    ) -> None:
        self.min_confidence = min_confidence

        font_path = Path(r"C:\Windows\Fonts\msyh.ttc")

        self._font = ImageFont.truetype(
            str(font_path),
            font_size,
        )

    def draw(
        self,
        image: np.ndarray,
        results: list[OCRResult],
    ) -> np.ndarray:
        annotated = image.copy()

        # OpenCV BGR -> Pillow RGB
        rgb_image = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)

        for result in results:
            if result.confidence < self.min_confidence:
                continue

            points = [
                (int(x), int(y))
                for x, y in result.bbox
            ]

            draw.line(
                points + [points[0]],
                fill=(0, 255, 0),
                width=2,
            )

            x, y = points[0]

            label = (
                f"{result.text} "
                f"({result.confidence:.2f})"
            )

            text_y = max(y - 22, 0)

            draw.text(
                (x, text_y),
                label,
                font=self._font,
                fill=(0, 255, 0),
            )

        # Pillow RGB -> OpenCV BGR
        result_image = cv2.cvtColor(
            np.asarray(pil_image),
            cv2.COLOR_RGB2BGR,
        )

        return result_image