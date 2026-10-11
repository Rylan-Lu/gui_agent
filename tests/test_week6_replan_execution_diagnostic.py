"""All tests are static: they NEVER start the desktop or load the VLM."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "robustness"))
from week6_real_e2e import safe_action_types, safe_judge_code, safe_step_outcomes, make_record, TrialSpec


def _result(reason, success=False):
    return SimpleNamespace(judge_result=SimpleNamespace(success=success, reason=reason))


def test_safe_judge_success():
    assert safe_judge_code(_result("anything", True)) == "passed"


def test_safe_judge_no_receipt():
    assert safe_judge_code(_result("local receipt missing")) == "receipt_missing"


def test_safe_judge_receipt_mismatch():
    assert safe_judge_code(_result("local receipt mismatch")) == "receipt_mismatch"


def test_safe_judge_visual_failure_sanitized():
    assert safe_judge_code(_result("expected text not found: PRIVATE_INFORMATION")) == "visual_marker_missing"


def test_safe_judge_missing_ocr():
    assert safe_judge_code(_result("no OCR results available")) == "ocr_unavailable"


def test_safe_judge_not_run():
    assert safe_judge_code(SimpleNamespace(judge_result=None, failure_decision=None)) == "not_run"


def test_safe_action_types_omit_text_and_description():
    plan = SimpleNamespace(steps=[
        SimpleNamespace(description="PRIVATE_DESCRIPTION", action=SimpleNamespace(
            action_type=SimpleNamespace(value="type_text"), text="PRIVATE_TEXT")),
        SimpleNamespace(description="PRIVATE_UI", action=SimpleNamespace(
            action_type=SimpleNamespace(value="click"), element="PRIVATE_ELEMENT")),
    ])
    assert safe_action_types(plan) == ["type_text", "click"]
    assert "PRIVATE" not in str(safe_action_types(plan))


def test_safe_step_outcomes_no_errors_leaked():
    runtime = SimpleNamespace(state=SimpleNamespace(history=[
        SimpleNamespace(success=False, description="PRIVATE", error="PRIVATE"),
        SimpleNamespace(success=True, description="PRIVATE", error=None),
        SimpleNamespace(success=None, description="PRIVATE", error=None),
    ]))
    assert safe_step_outcomes(runtime) == ["fail", "pass", "unknown"]


def test_record_accepts_instrumentation_fields():
    record = make_record(
        TrialSpec("text_entry", "enhanced", 3, "grounding_replan_window", "local-vlm"),
        success=False, human_success=False, duration_s=40.2, setup_duration_s=12,
        steps=4, failure_category="none", recovery_status="none", replan_calls=1,
        injected_count=2,
    )
    record.update(final_judge_code="visual_marker_missing", step_outcomes=["fail", "pass"])
    assert record["replan_calls"] == 1
    assert record["final_judge_code"] == "visual_marker_missing"
