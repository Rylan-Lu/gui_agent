from pathlib import Path

import cv2

from gui_agent.perception.screen_capture import ScreenCapture


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "ocr_benchmark.png"

    with ScreenCapture() as capture:
        frame = capture.capture()

    success = cv2.imwrite(
        str(output_path),
        frame.image,
    )

    if not success:
        raise RuntimeError("Failed to save screenshot.")

    print("Screenshot saved:")
    print(output_path.resolve())
    print("Image size:", frame.image_size)
    print("Region:", frame.region)


if __name__ == "__main__":
    main()