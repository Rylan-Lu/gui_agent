from typing import TypeAlias

Point : TypeAlias = tuple[float, float]
Region : TypeAlias = tuple[int, int, int, int]
ImageSize : TypeAlias = tuple[int, int]

class CoordinateMapper:
    @staticmethod
    def image_to_desktop(point: Point, *, image_size: ImageSize, region: Region) -> Point:
        x, y = point

        image_width, image_height = image_size

        left, top, region_width, region_height = region

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image size must be positive")

        if region_width <= 0 or region_height <= 0:
            raise ValueError("Region size must be positive")

        if not (0 <= x <= region_width and 0 <= y <= region_height):
            raise ValueError("Point must be within region")

        scale_x = region_width / image_width
        scale_y = region_height / image_height

        desktop_x = left + x * scale_x
        desktop_y = top + y * scale_y

        return desktop_x, desktop_y
