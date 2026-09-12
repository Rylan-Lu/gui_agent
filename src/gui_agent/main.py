from gui_agent.ocr.easyocr_engine import EasyOCR_ENGINE
from gui_agent.perception.screen_capture import ScreenCapture


def main() -> None:
    with ScreenCapture() as capture:
        image = capture.capture()

    engine = EasyOCR_ENGINE()
    results = engine.recognize(image)

    results = sorted(results, key=lambda result: result.confidence, reverse=True)

    for result in results:
        print(
            f"text={result.text!r}, "
            f"confidence={result.confidence:.3f}"
        )


if __name__ == "__main__":
    main()