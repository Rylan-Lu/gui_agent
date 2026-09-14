from pathlib import Path
from time import perf_counter

from paddleocr import PaddleOCR


IMAGE_PATH = (
    Path(r"C:\Users\28468\Desktop\gui_agent")
    / "scripts"
    / "outputs"
    / "ocr_benchmark.png"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    print("Loading PaddleOCR GPU...")

    ocr = PaddleOCR(
        lang="ch",
        device="gpu:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("Running OCR...")

    # GPU 首次调用可能包含初始化开销
    start = perf_counter()

    results = ocr.predict(
        str(IMAGE_PATH)
    )

    elapsed = perf_counter() - start

    count = 0

    for result in results:
        data = result.json["res"]

        texts = data["rec_texts"]
        scores = data["rec_scores"]

        for text, score in zip(
            texts,
            scores,
        ):
            count += 1

            print(
                f"{count:03d} | "
                f"{float(score):.3f} | "
                f"{text}"
            )

    print()
    print("Total detections:", count)
    print(f"GPU inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()