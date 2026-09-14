import time

from gui_agent.control.controller import Controller
from gui_agent.locator.coordinate_mapper import CoordinateMapper
from gui_agent.locator.ui_locator import UILocator
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine
from gui_agent.perception.screen_capture import ScreenCapture

TARGET_TEXT = "E2E_TARGET"
INPUT_TEXT = "GUI Agent 第二周 E2E 测试成功 2026"


def main() -> None:
    print("请在 5 秒内切换到 GUI Agent E2E Target 窗口...")
    time.sleep(5)

    with ScreenCapture() as capture:
        frame = capture.capture()

    print("Screenshot captured.")
    print("Image size:", frame.image_size)
    print("Region:", frame.region)

    engine = PaddleOCREngine(
        lang="ch",
        device="gpu:0",
        min_confidence=0.0,
    )
    results = engine.recognize(frame.image)
    print("OCR detections:", len(results))

    target = UILocator().find_one(
        results,
        TARGET_TEXT,
        mode="fuzzy",
        fuzzy_threshold=0.8,
    )

    print("Target:", target.result.text)
    print("OCR confidence:", target.result.confidence)
    print("Match score:", target.match_score)
    print("Image center:", target.image_center)

    desktop_center = CoordinateMapper.image_to_desktop(
        target.image_center,
        image_size=frame.image_size,
        region=frame.region,
    )
    print("Desktop center:", desktop_center)

    controller = Controller()
    controller.click(desktop_center, duration=0.4)
    time.sleep(0.5)
    controller.type_text(INPUT_TEXT)

    print("E2E action completed.")


if __name__ == "__main__":
    main()
