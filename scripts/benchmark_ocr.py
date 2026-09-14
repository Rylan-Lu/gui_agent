import argparse
from pathlib import Path
from time import perf_counter

import cv2

from gui_agent.ocr.base import OCREngine
from gui_agent.ocr.easyocr_engine import EasyOCREngine
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

DEFAULT_IMAGE = Path(__file__).resolve().parent / "outputs" / "ocr_benchmark.png"


def create_engine(name: str) -> OCREngine:
    if name == "paddle":
        return PaddleOCREngine(
            lang="ch",
            device="gpu:0",
            min_confidence=0.0,
        )
    if name == "easyocr":
        return EasyOCREngine()
    raise ValueError(f"Unsupported OCR engine: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=("paddle", "easyocr"),
        default="paddle",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {args.image}")

    print(f"Loading {args.engine}...")
    engine = create_engine(args.engine)

    start = perf_counter()
    results = engine.recognize(image)
    elapsed = perf_counter() - start

    for index, result in enumerate(results, start=1):
        print(
            f"{index:03d} | "
            f"{result.confidence:.3f} | "
            f"{result.text}"
        )

    print()
    print("Total detections:", len(results))
    print(f"Inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
