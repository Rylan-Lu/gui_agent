from typing import TypeAlias

from gui_agent.ocr.base import BoundingBox


Point: TypeAlias = tuple[float, float]
ImagePoint: TypeAlias = Point
DesktopPoint: TypeAlias = Point

Region: TypeAlias = tuple[int, int, int, int]
DesktopRegion: TypeAlias = Region

ImageSize: TypeAlias = tuple[int, int]


class CoordinateMapper:
    @staticmethod
    def image_to_desktop(
        point: ImagePoint,
        *,
        image_size: ImageSize,
        region: DesktopRegion,
    ) -> DesktopPoint:
        """
        Map an image-space point to desktop coordinates.

        Image points represent actionable positions such as mouse targets,
        therefore they use half-open image bounds:

            0 <= x < image_width
            0 <= y < image_height

        The desktop region is represented as:

            (left, top, width, height)

        The mapping supports both region offsets and image scaling.
        """
        CoordinateMapper._validate_sizes(image_size, region)

        image_width, image_height = image_size
        left, top, region_width, region_height = region
        x, y = point

        if not (0 <= x < image_width and 0 <= y < image_height):
            raise ValueError("Point must be within image")

        scale_x = region_width / image_width
        scale_y = region_height / image_height

        return (
            left + x * scale_x,
            top + y * scale_y,
        )

    @staticmethod
    def bbox_to_desktop(
        bbox: BoundingBox,
        *,
        image_size: ImageSize,
        region: DesktopRegion,
    ) -> BoundingBox:
        """
        Map an image-space bounding box to desktop coordinates.

        Bounding-box coordinates represent geometric edges rather than
        directly actionable pixel positions. Therefore points on the image
        boundary are valid:

            0 <= x <= image_width
            0 <= y <= image_height

        This allows a box covering the complete image to have its right and
        bottom edges exactly at image_width and image_height.
        """
        CoordinateMapper._validate_sizes(image_size, region)

        image_width, image_height = image_size
        left, top, region_width, region_height = region

        scale_x = region_width / image_width
        scale_y = region_height / image_height

        mapped_points: list[Point] = []

        for x, y in bbox:
            if not (0 <= x <= image_width and 0 <= y <= image_height):
                raise ValueError("Bounding box point must be within image")

            mapped_points.append(
                (
                    left + x * scale_x,
                    top + y * scale_y,
                )
            )

        return tuple(mapped_points)

    @staticmethod
    def _validate_sizes(
        image_size: ImageSize,
        region: DesktopRegion,
    ) -> None:
        """
        Validate image and desktop-region dimensions.

        Region origins may be negative because Windows multi-monitor desktop
        coordinates can extend to the left or above the primary monitor.
        Only width and height are required to be positive.
        """
        image_width, image_height = image_size
        _, _, region_width, region_height = region

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image size must be positive")

        if region_width <= 0 or region_height <= 0:
            raise ValueError("Region size must be positive")