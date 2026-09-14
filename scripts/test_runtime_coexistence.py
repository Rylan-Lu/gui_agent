from pathlib import Path

import torch
from paddleocr import PaddleOCR


IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "ocr_benchmark.png"
)


def torch_gpu_test(label: str) -> None:
    x = torch.randn(
        2048,
        2048,
        device="cuda",
    )

    y = x @ x
    torch.cuda.synchronize()

    print(
        f"{label}: PASS",
        y.device,
    )


def main() -> None:
    print("=== 1. Torch GPU before PaddleOCR ===")
    torch_gpu_test("Torch before PaddleOCR")

    print()
    print("=== 2. Initialize PaddleOCR CPU ===")

    ocr = PaddleOCR(
        lang="ch",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("PaddleOCR initialization: PASS")

    print()
    print("=== 3. PaddleOCR inference ===")

    results = ocr.predict(
        str(IMAGE_PATH)
    )

    count = 0

    for result in results:
        count += len(
            result.json["res"]["rec_texts"]
        )

    print(
        "PaddleOCR inference: PASS"
    )
    print(
        "Detections:",
        count,
    )

    print()
    print("=== 4. Torch GPU after PaddleOCR ===")

    torch_gpu_test(
        "Torch after PaddleOCR"
    )

    print()
    print("==============================")
    print("RUNTIME COEXISTENCE: PASS")
    print("==============================")


if __name__ == "__main__":
    main()