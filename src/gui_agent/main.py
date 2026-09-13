import cv2

from gui_agent.locator.coordinate_mapper import CoordinateMapper
from gui_agent.locator.ui_locator import UILocator
from gui_agent.locator.visualizer import LocatorVisualizer
from gui_agent.ocr.easyocr_engine import EasyOCR_ENGINE
from gui_agent.perception.screen_capture import ScreenCapture


def main() -> None:
    with ScreenCapture() as capture:
        frame = capture.capture()

    engine = EasyOCR_ENGINE()

    results = engine.recognize(
        frame.image
    )

    locator = UILocator()

    target = locator.find_one(
        results,
        "定时任务",
        mode="fuzzy",
        fuzzy_threshold=0.8,
    )

    desktop_center = CoordinateMapper.image_to_desktop(
        target.image_center,
        image_size=frame.image_size,
        region=frame.region,
    )

    desktop_bbox = CoordinateMapper.bbox_to_desktop(
        target.image_bbox,
        image_size=frame.image_size,
        region=frame.region,
    )

    print(
        "OCR text:",
        target.result.text,
    )

    print(
        "OCR confidence:",
        target.result.confidence,
    )

    print(
        "Match score:",
        target.match_score,
    )

    print(
        "Image bbox:",
        target.image_bbox,
    )

    print(
        "Image center:",
        target.image_center,
    )

    print(
        "Desktop bbox:",
        desktop_bbox,
    )

    print(
        "Desktop center:",
        desktop_center,
    )

    visualizer = LocatorVisualizer()

    annotated = visualizer.draw_target(
        frame.image,
        target,
    )

    cv2.imwrite(
        "locator_debug.png",
        annotated,
    )


if __name__ == "__main__":
    main()