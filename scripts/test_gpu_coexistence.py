from pathlib import Path

import torch
import torch.nn.functional as F

from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

IMAGE_PATH = Path(__file__).resolve().parent / "outputs" / "ocr_benchmark.png"


def torch_cudnn_test(label: str) -> None:
    x = torch.randn((8, 16, 128, 128), device="cuda")
    weight = torch.randn((32, 16, 3, 3), device="cuda")
    output = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()
    print(f"{label}: PASS {output.device}")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark image not found: {IMAGE_PATH}. "
            "Run scripts/save_ocr_benchmark.py first."
        )

    print("=== 1. Torch cuDNN before PaddleOCR ===")
    torch_cudnn_test("Torch before PaddleOCR")

    print("\n=== 2. PaddleOCR GPU ===")
    engine = PaddleOCREngine(device="gpu:0")

    import cv2

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Failed to load image: {IMAGE_PATH}")

    results = engine.recognize(image)
    print("PaddleOCR GPU: PASS")
    print("Detections:", len(results))

    print("\n=== 3. Torch cuDNN after PaddleOCR ===")
    torch_cudnn_test("Torch after PaddleOCR")

    print("\nGPU COEXISTENCE: PASS")


if __name__ == "__main__":
    main()
