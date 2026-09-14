from pathlib import Path
from time import perf_counter

from paddleocr import PaddleOCR


IMAGE_PATH = Path("outputs") / "ocr_benchmark.png"


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH.resolve()}"
        )

    ocr = PaddleOCR(
        lang="ch",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

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
        polygons = data["rec_polys"]

        for text, score, polygon in zip(
            texts,
            scores,
            polygons,
        ):
            count += 1

            print(
                f"{count:03d} | "
                f"{float(score):.3f} | "
                f"{text}"
            )

    print()
    print(f"Total detections: {count}")
    print(f"Inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()