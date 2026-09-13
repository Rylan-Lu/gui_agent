from pydoc import describe
from typing import TypeAlias
from gui_agent.ocr.base import BoundingBox

Point : TypeAlias = tuple[float, float]
Region : TypeAlias = tuple[int, int, int, int]
ImageSize : TypeAlias = tuple[int, int]

class CoordinateMapper:
    @staticmethod
    def image_to_desktop(point: Point, *, image_size: ImageSize, region: Region) -> Point:
        image_width, image_height = image_size
        left, top, region_width, region_height = region

        CoordinateMapper._validate_sizes(image_size, region)

        x, y = point

        if not (0 <= x <= image_width and 0 <= y <= image_height):
            raise ValueError("Point must be within image")

        scale_x = region_width / image_width
        scale_y = region_height / image_height

        desktop_x = left + x * scale_x
        desktop_y = top + y * scale_y

        return desktop_x, desktop_y

    @staticmethod
    def bbox_to_desktop(bbox: BoundingBox, *, image_size: ImageSize, region: Region) -> BoundingBox:
        image_width, image_height = image_size
        left, top, region_width, region_height = region

        CoordinateMapper._validate_sizes(image_size, region)

        scale_x = region_width / image_width
        scale_y = region_height / image_height

        mapped_points = []

        for x, y in bbox:
            if not (0 <= x <= region_width and 0 <= y <= region_height):
                raise ValueError("Point must be within region")

            desktop_x = left + x * scale_x
            desktop_y = top + y * scale_y

            mapped_points.append((desktop_x, desktop_y))

        return tuple(mapped_points)

    @staticmethod
    def _validate_sizes(image_size: ImageSize, region: Region) -> None:
        image_width, image_height = image_size
        _, _, region_width, region_height = region

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image size must be positive")

        if region_width <= 0 or region_height <= 0:
            raise ValueError("Region size must be positive")
