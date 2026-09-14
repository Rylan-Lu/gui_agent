from pathlib import Path

import cv2

from gui_agent.perception.screen_capture import ScreenCapture

OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "ocr_benchmark.png"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ScreenCapture() as capture:
        frame = capture.capture()

    if not cv2.imwrite(str(OUTPUT_PATH), frame.image):
        raise RuntimeError("Failed to save screenshot")

    print("Screenshot saved:", OUTPUT_PATH)
    print("Image size:", frame.image_size)
    print("Region:", frame.region)


if __name__ == "__main__":
    main()
