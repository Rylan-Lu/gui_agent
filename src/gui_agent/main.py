from gui_agent.locator.ui_locator import (
    AmbiguousTargetError,
    TargetNotFoundError,
    UILocator,
)
from gui_agent.ocr.easyocr_engine import EasyOCR_ENGINE
from gui_agent.perception.screen_capture import ScreenCapture


def main() -> None:
    with ScreenCapture() as capture:
        image = capture.capture()

    engine = EasyOCR_ENGINE()
    results = engine.recognize(image)

    locator = UILocator()

    try:
        target = locator.find_one(
            results,
            "lasdjasldjasldjasld",
        )

        print(f"Found: {target.result.text}")
        print(
            f"Confidence: "
            f"{target.result.confidence:.3f}"
        )
        print(
            f"Image center: "
            f"{target.image_center}"
        )

    except TargetNotFoundError as exc:
        print(exc)

    except AmbiguousTargetError as exc:
        print(exc)


if __name__ == "__main__":
    main()