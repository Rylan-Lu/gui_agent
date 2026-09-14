import pytest

from gui_agent.locator.ui_locator import (
    AmbiguousTargetError,
    TargetNotFoundError,
    UILocator,
    normalize_text,
)
from gui_agent.ocr.base import OCRResult


def ocr(text: str, confidence: float = 0.9, *, x: float = 0, y: float = 0, w: float = 100, h: float = 40) -> OCRResult:
    return OCRResult(
        text=text,
        confidence=confidence,
        bbox=((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
    )


NORMALIZATION_CASES = [
    ("Settings", "settings"),
    ("SETTINGS", "settings"),
    (" settings ", "settings"),
    ("  Settings   Panel ", "settings panel"),
    ("Settings\nPanel", "settings panel"),
    ("Settings\tPanel", "settings panel"),
    ('"Settings"', "settings"),
    ("'Settings'", "settings"),
    ("“Settings”", "settings"),
    ("‘Settings’", "settings"),
    ('  "Settings"  ', "settings"),
    ('"“Settings”"', "settings"),
    ("ＡＢＣ", "abc"),
    ("１２３", "123"),
    ("Ｅ２Ｅ＿ＴＡＲＧＥＴ", "e2e_target"),
    ("ｓｅｔｔｉｎｇｓ", "settings"),
    ("Café", "café"),
    ("Straße", "strasse"),
    ("中文 测试", "中文 测试"),
    (" 中文   测试 ", "中文 测试"),
    ("SAVE_AS", "save_as"),
    ("Open File", "open file"),
    ("\n\t", ""),
    ("", ""),
    ("  A  B  C  ", "a b c"),
    ('"A B C"', "a b c"),
    ("“中文按钮”", "中文按钮"),
    ("１．２．３", "1.2.3"),
    ("Ｆｉｌｅ－Ｎａｍｅ", "file-name"),
    ("KELVIN", "kelvin"),
]


@pytest.mark.parametrize(("raw", "expected"), NORMALIZATION_CASES)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


EXACT_CASES = [
    ("Settings", "settings", True),
    (" SETTINGS ", "settings", True),
    ('"Settings"', "settings", True),
    ("ＡＢＣ", "abc", True),
    ("中文 按钮", "中文   按钮", True),
    ("Settings", "Setting", False),
    ("Open", "Open File", False),
    ("Save", "Save As", False),
    ("E2E_TARGET", "E2E TARGET", False),
    ("abc", "abcd", False),
    ("", "", True),
    ("A", "a", True),
]


@pytest.mark.parametrize(("target", "candidate", "expected"), EXACT_CASES)
def test_exact_matching(target, candidate, expected):
    matches = UILocator().find_all([ocr(candidate)], target, mode="exact")
    assert bool(matches) is expected


FUZZY_CASES = [
    ("settings", "setting", 0.80, True),
    ("settings", "sett1ngs", 0.80, True),
    ("settings", "preferences", 0.80, False),
    ("e2e_target", "e2e-target", 0.80, True),
    ("open file", "open flle", 0.80, True),
    ("save as", "save", 0.80, False),
    ("中文按钮", "中文按纽", 0.75, True),
    ("project", "projct", 0.80, True),
    ("cancel", "cance1", 0.80, True),
    ("submit", "sub", 0.80, False),
]


@pytest.mark.parametrize(("target", "candidate", "threshold", "expected"), FUZZY_CASES)
def test_fuzzy_matching(target, candidate, threshold, expected):
    matches = UILocator().find_all(
        [ocr(candidate)],
        target,
        mode="fuzzy",
        fuzzy_threshold=threshold,
    )
    assert bool(matches) is expected


@pytest.mark.parametrize(
    ("confidence", "minimum", "expected"),
    [
        (0.0, 0.0, True),
        (0.49, 0.5, False),
        (0.5, 0.5, True),
        (0.79, 0.8, False),
        (0.8, 0.8, True),
        (1.0, 1.0, True),
    ],
)
def test_confidence_filter(confidence, minimum, expected):
    matches = UILocator().find_all(
        [ocr("Settings", confidence)],
        "Settings",
        min_confidence=minimum,
    )
    assert bool(matches) is expected


@pytest.mark.parametrize(
    ("x", "y", "roi", "expected"),
    [
        (0, 0, (0, 0, 100, 100), True),
        (49, 49, (0, 0, 100, 100), True),
        (50, 50, (0, 0, 100, 100), True),
        (99, 99, (0, 0, 100, 100), False),
        (100, 100, (100, 100, 100, 100), True),
        (149, 149, (100, 100, 100, 100), True),
        (150, 150, (100, 100, 100, 100), True),
        (199, 199, (100, 100, 100, 100), False),
    ],
)
def test_roi_filter_uses_center(x, y, roi, expected):
    # 2x2 box -> center is x+1, y+1.
    result = ocr("Settings", x=x, y=y, w=2, h=2)
    matches = UILocator().find_all([result], "Settings", roi=roi)
    assert bool(matches) is expected


def test_find_all_sorts_by_match_score_then_confidence():
    results = [
        ocr("sett1ngs", 0.99),
        ocr("settings", 0.60),
        ocr("settings", 0.95),
    ]
    matches = UILocator().find_all(
        results,
        "settings",
        mode="fuzzy",
        fuzzy_threshold=0.5,
    )
    assert [item.result.confidence for item in matches[:2]] == [0.95, 0.60]
    assert matches[0].match_score == 1.0


def test_find_one_returns_unique_match():
    found = UILocator().find_one([ocr("Settings")], "Settings")
    assert found.result.text == "Settings"
    assert found.image_center == (50.0, 20.0)


def test_find_one_raises_when_missing():
    with pytest.raises(TargetNotFoundError):
        UILocator().find_one([ocr("Other")], "Settings")


def test_find_one_raises_when_ambiguous():
    with pytest.raises(AmbiguousTargetError):
        UILocator().find_one([ocr("Settings"), ocr("settings")], "Settings")


@pytest.mark.parametrize("value", [-0.1, -1.0, 1.1, 2.0])
def test_invalid_min_confidence(value):
    with pytest.raises(ValueError):
        UILocator().find_all([], "x", min_confidence=value)


@pytest.mark.parametrize("value", [-0.1, -1.0, 1.1, 2.0])
def test_invalid_fuzzy_threshold(value):
    with pytest.raises(ValueError):
        UILocator().find_all([], "x", fuzzy_threshold=value)


@pytest.mark.parametrize("roi", [(0, 0, 0, 10), (0, 0, 10, 0), (0, 0, -1, 10), (0, 0, 10, -1)])
def test_invalid_roi_dimensions(roi):
    with pytest.raises(ValueError):
        UILocator().find_all([ocr("x")], "x", roi=roi)


def test_invalid_match_mode():
    with pytest.raises(ValueError):
        UILocator().find_all([ocr("x")], "x", mode="contains")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        (((0, 0), (10, 0), (10, 10), (0, 10)), (5.0, 5.0)),
        (((10, 20), (30, 20), (30, 60), (10, 60)), (20.0, 40.0)),
        (((-10, -10), (10, -10), (10, 10), (-10, 10)), (0.0, 0.0)),
        (((0.5, 1.5), (2.5, 1.5), (2.5, 5.5), (0.5, 5.5)), (1.5, 3.5)),
    ],
)
def test_calculate_center(bbox, expected):
    assert UILocator._calculate_center(OCRResult("x", 1.0, bbox)) == expected
