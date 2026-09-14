import sys
import types

import numpy as np
import pytest

from gui_agent.ocr.easyocr_engine import EasyOCREngine
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine


class FakePrediction:
    def __init__(self, data):
        self.json = {"res": data}


class FakePaddleBackend:
    def __init__(self, predictions):
        self.predictions = predictions
        self.last_image = None

    def predict(self, image):
        self.last_image = image
        return self.predictions


def make_paddle_engine(data, min_confidence=0.0):
    engine = object.__new__(PaddleOCREngine)
    engine.min_confidence = min_confidence
    engine._ocr = FakePaddleBackend([FakePrediction(data)])
    return engine


def test_paddle_recognize_converts_boxes():
    engine = make_paddle_engine(
        {
            "rec_texts": ["Settings"],
            "rec_scores": [0.95],
            "rec_boxes": [[10, 20, 30, 40]],
        }
    )
    results = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert results[0].text == "Settings"
    assert results[0].confidence == pytest.approx(0.95)
    assert results[0].bbox == ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0))


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Settings", "Settings"),
        (" Settings ", "Settings"),
        ("\nSettings\t", "Settings"),
        ("中文按钮", "中文按钮"),
        (123, "123"),
    ],
)
def test_paddle_text_conversion(raw_text, expected):
    engine = make_paddle_engine(
        {
            "rec_texts": [raw_text],
            "rec_scores": [1.0],
            "rec_boxes": [[0, 0, 10, 10]],
        }
    )
    results = engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8))
    assert results[0].text == expected


@pytest.mark.parametrize(
    ("score", "minimum", "expected_count"),
    [
        (0.0, 0.0, 1),
        (0.49, 0.5, 0),
        (0.5, 0.5, 1),
        (0.79, 0.8, 0),
        (0.8, 0.8, 1),
        (1.0, 1.0, 1),
    ],
)
def test_paddle_confidence_filter(score, minimum, expected_count):
    engine = make_paddle_engine(
        {
            "rec_texts": ["x"],
            "rec_scores": [score],
            "rec_boxes": [[0, 0, 10, 10]],
        },
        min_confidence=minimum,
    )
    assert len(engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8))) == expected_count


@pytest.mark.parametrize("text", ["", " ", "\n", "\t"])
def test_paddle_skips_blank_text(text):
    engine = make_paddle_engine(
        {
            "rec_texts": [text],
            "rec_scores": [1.0],
            "rec_boxes": [[0, 0, 10, 10]],
        }
    )
    assert engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize("box", [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5], [[1, 2], [3, 4]]])
def test_paddle_skips_invalid_box_shape(box):
    engine = make_paddle_engine(
        {
            "rec_texts": ["x"],
            "rec_scores": [1.0],
            "rec_boxes": [box],
        }
    )
    # [[1,2],[3,4]] flattens to 4 and is intentionally supported.
    count = len(engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8)))
    assert count == (1 if np.asarray(box).size == 4 else 0)


def test_paddle_handles_multiple_predictions():
    engine = object.__new__(PaddleOCREngine)
    engine.min_confidence = 0.0
    engine._ocr = FakePaddleBackend(
        [
            FakePrediction({"rec_texts": ["A"], "rec_scores": [1], "rec_boxes": [[0, 0, 1, 1]]}),
            FakePrediction({"rec_texts": ["B"], "rec_scores": [1], "rec_boxes": [[1, 1, 2, 2]]}),
        ]
    )
    assert [r.text for r in engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8))] == ["A", "B"]


def test_paddle_zip_uses_shortest_sequence():
    engine = make_paddle_engine(
        {
            "rec_texts": ["A", "B"],
            "rec_scores": [1.0],
            "rec_boxes": [[0, 0, 1, 1], [1, 1, 2, 2]],
        }
    )
    assert len(engine.recognize(np.zeros((2, 2, 3), dtype=np.uint8))) == 1


@pytest.mark.parametrize("image", [None, np.zeros((10, 10)), np.zeros((10, 10, 3, 1))])
def test_paddle_invalid_images(image):
    engine = make_paddle_engine({})
    with pytest.raises(ValueError):
        engine.recognize(image)  # type: ignore[arg-type]


@pytest.mark.parametrize("minimum", [-1.0, -0.1, 1.1, 2.0])
def test_paddle_init_rejects_invalid_min_confidence(minimum):
    with pytest.raises(ValueError):
        PaddleOCREngine(min_confidence=minimum)


def test_paddle_init_cpu_does_not_initialize_torch(monkeypatch):
    import gui_agent.ocr.paddleocr_engine as module

    calls = []
    monkeypatch.setattr(module, "initialize_torch_gpu_runtime", lambda: calls.append("torch"))

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = lambda **kwargs: ("ocr", kwargs)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    engine = PaddleOCREngine(device="cpu")
    assert calls == []
    assert engine._ocr[1]["device"] == "cpu"


def test_paddle_init_gpu_initializes_torch_first(monkeypatch):
    import gui_agent.ocr.paddleocr_engine as module

    calls = []
    monkeypatch.setattr(module, "initialize_torch_gpu_runtime", lambda: calls.append("torch"))

    fake_module = types.ModuleType("paddleocr")

    def fake_ctor(**kwargs):
        calls.append("paddle")
        return kwargs

    fake_module.PaddleOCR = fake_ctor
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    PaddleOCREngine(device="gpu:0")
    assert calls == ["torch", "paddle"]


def test_easyocr_recognize_converts_results():
    engine = object.__new__(EasyOCREngine)

    class Reader:
        def readtext(self, image):
            return [
                ([[1, 2], [3, 2], [3, 4], [1, 4]], "ABC", 0.75),
                ([[10, 20], [30, 20], [30, 40], [10, 40]], "中文", 0.9),
            ]

    engine._reader = Reader()
    results = engine.recognize(np.zeros((5, 5, 3), dtype=np.uint8))
    assert [r.text for r in results] == ["ABC", "中文"]
    assert results[0].bbox[0] == (1.0, 2.0)
    assert results[1].confidence == pytest.approx(0.9)


def test_easyocr_init_uses_expected_languages_and_gpu(monkeypatch):
    captured = {}
    fake_module = types.ModuleType("easyocr")

    class Reader:
        def __init__(self, languages, gpu):
            captured["languages"] = languages
            captured["gpu"] = gpu

    fake_module.Reader = Reader
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)
    EasyOCREngine()
    assert captured == {"languages": ["ch_sim", "en"], "gpu": True}
