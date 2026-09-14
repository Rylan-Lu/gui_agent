import cv2
import numpy as np

from gui_agent.locator.ui_locator import LocatedElement


class LocatorVisualizer:
    def draw_target(
        self,
        image: np.ndarray,
        target: LocatedElement,
    ) -> np.ndarray:
        annotated = image.copy()
        points = np.array(
            [(round(x), round(y)) for x, y in target.image_bbox],
            dtype=np.int32,
        )

        cv2.polylines(
            annotated,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
        )

        center = (
            round(target.image_center[0]),
            round(target.image_center[1]),
        )
        cv2.circle(
            annotated,
            center,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

        label = (
            f"{target.result.text} | "
            f"match={target.match_score:.2f} | "
            f"ocr={target.result.confidence:.2f}"
        )
        label_x = int(points[:, 0].min())
        label_y = max(int(points[:, 1].min()) - 10, 20)

        cv2.putText(
            annotated,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated
