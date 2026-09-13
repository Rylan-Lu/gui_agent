from gui_agent.control.controller import Controller
from gui_agent.locator.coordinate_mapper import CoordinateMapper
from gui_agent.locator.ui_locator import UILocator
from gui_agent.ocr.easyocr_engine import EasyOCR_ENGINE
from gui_agent.perception.screen_capture import ScreenCapture


TARGET = "开始第二周任务"


def main() -> None:
    # 1. 截图
    with ScreenCapture() as capture:
        frame = capture.capture()

    # 2. OCR
    engine = EasyOCR_ENGINE()
    results = engine.recognize(frame.image)

    # 3. 找目标
    locator = UILocator()

    target = locator.find_one(
        results,
        TARGET,
        mode="fuzzy",
        fuzzy_threshold=0.8,
    )

    print("OCR text:", target.result.text)
    print("OCR confidence:", target.result.confidence)
    print("Match score:", target.match_score)
    print("Image center:", target.image_center)

    # 4. 转换成桌面坐标
    desktop_center = CoordinateMapper.image_to_desktop(
        target.image_center,
        image_size=frame.image_size,
        region=frame.region,
    )

    print("Desktop center:", desktop_center)

    # 5. 只移动，不点击
    controller = Controller()

    print("Controller screen:", controller.screen_size)

    controller.move_to(
        desktop_center,
        duration=1.0,
    )


if __name__ == "__main__":
    main()