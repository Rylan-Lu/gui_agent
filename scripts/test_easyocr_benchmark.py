from pathlib import Path
from time import perf_counter

import cv2

from gui_agent.ocr.easyocr_engine import EasyOCR_ENGINE


IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "ocr_benchmark.png"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Failed to load benchmark image.")

    print("Loading EasyOCR...")

    engine = EasyOCR_ENGINE()

    print("Running OCR...")

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