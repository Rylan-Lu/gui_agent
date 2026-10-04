import pytest

from gui_agent.locator.coordinate_mapper import CoordinateMapper


POINT_CASES = [
    # 1:1 mapping
    ((0, 0), (100, 100), (0, 0, 100, 100), (0, 0)),
    ((50, 50), (100, 100), (0, 0, 100, 100), (50, 50)),
    ((99, 99), (100, 100), (0, 0, 100, 100), (99, 99)),

    # Region offset
    ((25, 75), (100, 100), (10, 20, 100, 100), (35, 95)),

    # Non-uniform scaling
    ((0, 0), (100, 100), (10, 20, 200, 300), (10, 20)),
    ((50, 50), (100, 100), (10, 20, 200, 300), (110, 170)),
    ((99, 99), (100, 100), (10, 20, 200, 300), (208, 317)),

    # Down-scaling
    ((100, 50), (200, 100), (0, 0, 100, 100), (50, 50)),

    # Up-scaling with offset
    ((199, 99), (200, 100), (100, 200, 400, 200), (498, 398)),

    # Typical desktop resolutions
    ((1280, 720), (2560, 1440), (0, 0, 2560, 1440), (1280, 720)),
    ((640, 360), (1280, 720), (0, 0, 2560, 1440), (1280, 720)),
    ((320, 180), (640, 360), (100, 50, 1280, 720), (740, 410)),

    # Floating-point image coordinates
    ((1.5, 2.5), (10, 10), (5, 6, 20, 30), (8, 13.5)),

    # Negative desktop origin is valid
    ((0, 0), (1, 1), (-100, -50, 2, 4), (-100, -50)),
]


@pytest.mark.parametrize(
    ("point", "image_size", "region", "expected"),
    POINT_CASES,
)
def test_image_to_desktop(point, image_size, region, expected):
    actual = CoordinateMapper.image_to_desktop(
        point,
        image_size=image_size,
        region=region,
    )

    assert actual == pytest.approx(expected)


BBOX_CASES = [
    # Full-image bounding box.
    # Unlike actionable points, geometric bbox edges may equal width/height.
    (
        ((0, 0), (100, 0), (100, 100), (0, 100)),
        (100, 100),
        (0, 0, 100, 100),
        ((0, 0), (100, 0), (100, 100), (0, 100)),
    ),

    # Region offset + non-uniform scaling
    (
        ((10, 20), (30, 20), (30, 40), (10, 40)),
        (100, 100),
        (100, 200, 200, 300),
        ((120, 260), (160, 260), (160, 320), (120, 320)),
    ),

    # Uniform scaling
    (
        ((0, 0), (50, 0), (50, 25), (0, 25)),
        (100, 50),
        (10, 20, 200, 100),
        ((10, 20), (110, 20), (110, 70), (10, 70)),
    ),

    # Typical desktop scaling
    (
        ((320, 180), (640, 180), (640, 360), (320, 360)),
        (1280, 720),
        (0, 0, 2560, 1440),
        ((640, 360), (1280, 360), (1280, 720), (640, 720)),
    ),

    # Floating-point bbox
    (
        ((0.5, 1.5), (2.5, 1.5), (2.5, 3.5), (0.5, 3.5)),
        (10, 10),
        (0, 0, 20, 40),
        ((1, 6), (5, 6), (5, 14), (1, 14)),
    ),
]


@pytest.mark.parametrize(
    ("bbox", "image_size", "region", "expected"),
    BBOX_CASES,
)
def test_bbox_to_desktop(bbox, image_size, region, expected):
    actual = CoordinateMapper.bbox_to_desktop(
        bbox,
        image_size=image_size,
        region=region,
    )

    for actual_point, expected_point in zip(actual, expected):
        assert actual_point == pytest.approx(expected_point)


@pytest.mark.parametrize(
    ("point", "image_size", "region"),
    [
        # Negative coordinates
        ((-1, 0), (100, 100), (0, 0, 100, 100)),
        ((0, -1), (100, 100), (0, 0, 100, 100)),

        # Right/bottom boundary itself is not an actionable image point
        ((100, 0), (100, 100), (0, 0, 100, 100)),
        ((0, 100), (100, 100), (0, 0, 100, 100)),
        ((100, 100), (100, 100), (0, 0, 100, 100)),

        # Beyond image boundary
        ((101, 0), (100, 100), (0, 0, 100, 100)),
        ((0, 101), (100, 100), (0, 0, 100, 100)),

        # Floating-point boundary violations
        ((-0.01, 50), (100, 100), (0, 0, 100, 100)),
        ((50, -0.01), (100, 100), (0, 0, 100, 100)),
        ((100.01, 50), (100, 100), (0, 0, 100, 100)),
        ((50, 100.01), (100, 100), (0, 0, 100, 100)),
    ],
)
def test_image_to_desktop_rejects_out_of_bounds(
    point,
    image_size,
    region,
):
    with pytest.raises(ValueError, match="Point must be within image"):
        CoordinateMapper.image_to_desktop(
            point,
            image_size=image_size,
            region=region,
        )


@pytest.mark.parametrize(
    ("image_size", "region"),
    [
        # Invalid image sizes
        ((0, 100), (0, 0, 100, 100)),
        ((100, 0), (0, 0, 100, 100)),
        ((-1, 100), (0, 0, 100, 100)),
        ((100, -1), (0, 0, 100, 100)),

        # Invalid region sizes
        ((100, 100), (0, 0, 0, 100)),
        ((100, 100), (0, 0, 100, 0)),
        ((100, 100), (0, 0, -1, 100)),
        ((100, 100), (0, 0, 100, -1)),
    ],
)
def test_invalid_sizes(image_size, region):
    with pytest.raises(ValueError):
        CoordinateMapper.image_to_desktop(
            (0, 0),
            image_size=image_size,
            region=region,
        )


@pytest.mark.parametrize(
    "bad_point",
    [
        (-1, 10),
        (10, -1),
        (101, 10),
        (10, 101),
    ],
)
def test_bbox_rejects_bad_point(bad_point):
    bbox = (
        (0, 0),
        (100, 0),
        bad_point,
        (0, 100),
    )

    with pytest.raises(
        ValueError,
        match="Bounding box point must be within image",
    ):
        CoordinateMapper.bbox_to_desktop(
            bbox,
            image_size=(100, 100),
            region=(0, 0, 100, 100),
        )


def test_bbox_allows_points_on_image_boundary():
    bbox = (
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    )

    actual = CoordinateMapper.bbox_to_desktop(
        bbox,
        image_size=(100, 100),
        region=(0, 0, 200, 300),
    )

    expected = (
        (0, 0),
        (200, 0),
        (200, 300),
        (0, 300),
    )

    for actual_point, expected_point in zip(actual, expected):
        assert actual_point == pytest.approx(expected_point)


def test_negative_desktop_region_origin_is_supported():
    actual = CoordinateMapper.image_to_desktop(
        (50, 25),
        image_size=(100, 50),
        region=(-1920, -100, 1920, 1080),
    )

    assert actual == pytest.approx((-960, 440))