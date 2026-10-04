import numpy as np
import pytest

from gui_agent.agent.observation import Observation
from gui_agent.agent.success_judge import TextSuccessJudge
from gui_agent.ocr.base import OCRResult


def make_observation(results):
    return Observation(
        screenshot=np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        ),
        screen_width=100,
        screen_height=100,
        ocr_result=results,
    )


def make_ocr(
    text,
    confidence=0.98,
    bbox=((0, 0), (10, 0), (10, 10), (0, 10)),
):
    return OCRResult(
        text=text,
        confidence=confidence,
        bbox=bbox,
    )


def test_success_when_text_found():
    observation = make_observation(
        [
            make_ocr(
                "Week4 GUI Agent E2E Test"
            )
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "Week4 GUI Agent E2E Test",
    )

    assert result.success is True
    assert result.reason == "expected text found"
    assert (
        result.matched_text
        == "Week4 GUI Agent E2E Test"
    )


def test_case_insensitive():
    observation = make_observation(
        [
            make_ocr(
                "NOTEPAD",
                confidence=0.95,
            )
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "notepad",
    )

    assert result.success is True


def test_success_when_text_split_across_results():
    observation = make_observation(
        [
            make_ocr("Week4 GUI"),
            make_ocr("Agent E2E Test"),
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "Week4 GUI Agent E2E Test",
    )

    assert result.success is True
    assert (
        result.reason
        == "expected text found across OCR results"
    )
    assert (
        result.matched_text
        == "Week4 GUI Agent E2E Test"
    )


def test_failure_when_text_missing():
    observation = make_observation(
        [
            make_ocr(
                "Something else",
                confidence=0.95,
            )
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "Notepad",
    )

    assert result.success is False
    assert result.matched_text is None


def test_failure_without_ocr():
    observation = make_observation(None)

    result = TextSuccessJudge().judge(
        observation,
        "Notepad",
    )

    assert result.success is False
    assert result.reason == "no OCR results available"


def test_empty_expected_text_rejected():
    observation = make_observation([])

    with pytest.raises(ValueError):
        TextSuccessJudge().judge(
            observation,
            "",
        )

def test_success_with_missing_underscores():
    observation = make_observation(
        [
            make_ocr(
                "GUI AGENT WEEK4 1790428383"
            )
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "GUI_AGENT_WEEK4_1790428383",
    )

    assert result.success is True


def test_success_compact_across_results():
    observation = make_observation(
        [
            make_ocr("WEEK4 FILE"),
            make_ocr("OPEN PASS"),
        ]
    )

    result = TextSuccessJudge().judge(
        observation,
        "WEEK4_FILE_OPEN_PASS",
    )

    assert result.success is True

def test_absent_judge_success_when_text_is_missing():
    from gui_agent.agent.success_judge import TextAbsentJudge

    observation = make_observation([make_ocr("Other text")])

    result = TextAbsentJudge().judge(
        observation,
        "Notepad",
    )

    assert result.success is True
    assert result.reason == "forbidden text is absent"


def test_absent_judge_failure_when_text_is_visible():
    from gui_agent.agent.success_judge import TextAbsentJudge

    observation = make_observation([make_ocr("Notepad")])

    result = TextAbsentJudge().judge(
        observation,
        "Notepad",
    )

    assert result.success is False
    assert result.matched_text == "Notepad"


def test_absent_judge_does_not_treat_missing_ocr_as_success():
    from gui_agent.agent.success_judge import TextAbsentJudge

    observation = make_observation(None)

    result = TextAbsentJudge().judge(
        observation,
        "Notepad",
    )

    assert result.success is False
    assert result.reason == "no OCR results available"


def test_absent_judge_compact_normalization_detects_visible_text():
    from gui_agent.agent.success_judge import TextAbsentJudge

    observation = make_observation(
        [make_ocr("GUI AGENT WEEK4 1790428383")]
    )

    result = TextAbsentJudge().judge(
        observation,
        "GUI_AGENT_WEEK4_1790428383",
    )

    assert result.success is False
