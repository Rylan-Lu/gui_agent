"""Platform-independent tests: never launch the desktop or OCR engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "robustness"
sys.path.insert(0, str(SCRIPTS))

from week6_real_e2e import (  # noqa: E402
    TrialSpec, ask_human_review, append_record, build_fixture, main, make_record,
)
from week6_real_summary import compare_modes, load_jsonl, percentile, save_report, summarize  # noqa: E402


@pytest.mark.parametrize("task,count", [
    ("browser_open", 5), ("browser_search", 8), ("text_entry", 4)
])
def test_trial_fixtures_have_executable_steps(tmp_path, task, count):
    fixture = build_fixture(TrialSpec(task, "baseline", 1), tmp_path)
    steps = fixture.plan_data["steps"]
    assert len(steps) == count
    assert [step["step_id"] for step in steps] == list(range(1, count + 1))
    assert all(step["action"] is not None for step in steps)


def test_local_search_fixture_prevents_premature_visible_success(tmp_path):
    fixture = build_fixture(TrialSpec("browser_search", "enhanced", 2), tmp_path)
    page = fixture.html_file.read_text(encoding="utf-8")
    assert "SEARCH_QUERY" in page
    assert 'id="result"></h2>' in page
    assert 'textContent=' in page
    assert fixture.marker in page  # only in JS; not displayed before submit
    assert fixture.query in page


def test_browser_open_fixture_is_local(tmp_path):
    fixture = build_fixture(TrialSpec("browser_open", "baseline", 1), tmp_path)
    assert fixture.html_file.is_file()
    text = fixture.html_file.read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text
    assert fixture.marker in text


def test_text_entry_fixture_uses_owned_receipt(tmp_path):
    fixture = build_fixture(TrialSpec("text_entry", "enhanced", 3), tmp_path)
    assert fixture.receipt.parent == tmp_path
    assert any(s["action"].get("element") == "E2E_TARGET" for s in fixture.plan_data["steps"])


@pytest.mark.parametrize("task,mode,trial,fault,backend", [
    ("bad", "baseline", 1, "none", "none"),
    ("browser_open", "bad", 1, "none", "none"),
    ("text_entry", "baseline", 0, "none", "none"),
    ("browser_open", "enhanced", 1, "grounding_once", "none"),
    ("text_entry", "baseline", 1, "none", "local-vlm"),
])
def test_invalid_trial_rejected(task, mode, trial, fault, backend):
    with pytest.raises(ValueError):
        TrialSpec(task, mode, trial, fault, backend)


def test_append_read_real_record_without_pii(tmp_path):
    spec = TrialSpec("text_entry", "enhanced", 1, "grounding_once")
    result = make_record(
        spec, success=True, human_success=None, duration_s=5.1,
        setup_duration_s=1.2, steps=4, failure_category="none",
        recovery_status="none", replan_calls=0, injected_count=1,
    )
    assert result["human_success"] is None
    assert "OCR" not in json.dumps(result)
    path = tmp_path / "real_runs.jsonl"
    append_record(path, result)
    assert load_jsonl(path) == [result]


def test_no_execution_without_explicit_execute(capsys):
    code = main(["--task", "text_entry", "--mode", "baseline", "--trial", "1"])
    assert code == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_explicit_human_unknown_stays_unknown():
    assert ask_human_review("unknown") is None
    assert ask_human_review("yes") is True
    assert ask_human_review("no") is False


def test_reviewer_input_can_remain_unknown(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "u")
    assert ask_human_review("prompt") is None


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 3, 5, 7, 9], .95) == pytest.approx(8.6)


def test_real_summary_separates_injected_from_natural():
    a = make_record(
        TrialSpec("text_entry", "baseline", 1, "none"), success=True,
        human_success=None, duration_s=1.0, setup_duration_s=0.0,
        steps=2, failure_category="none", recovery_status="none",
        replan_calls=0, injected_count=0,
    )
    b = make_record(
        TrialSpec("text_entry", "enhanced", 1, "grounding_once"), success=False,
        human_success=True, duration_s=2.0, setup_duration_s=0.0,
        steps=3, failure_category="grounding", recovery_status="unresolved",
        replan_calls=0, injected_count=1,
    )
    groups = summarize([a, b, {**a, "source": "MOCK-ONLY"}])
    assert len(groups) == 2
    by_fault = {r["fault"]: r for r in groups}
    assert by_fault["none"]["fully_reviewed"] == 0
    assert by_fault["grounding_once"]["fully_reviewed"] == 1
    assert by_fault["grounding_once"]["judge_false_negative"] == 1


def test_summarizer_outputs_csv_json(tmp_path):
    path = tmp_path / "real_runs.jsonl"
    row = make_record(
        TrialSpec("browser_open", "baseline", 1), success=True,
        human_success=True, duration_s=2.4, setup_duration_s=1.0,
        steps=5, failure_category="none", recovery_status="none",
        replan_calls=0, injected_count=0,
    )
    append_record(path, row)
    result = save_report(path, tmp_path)
    assert result[0]["runs"] == 1
    assert result[0]["sufficient_repetitions"] is False
    assert (tmp_path / "real_summary.csv").is_file()
    assert json.loads((tmp_path / "real_summary.json").read_text(encoding="utf-8"))["data_type"] == "REAL_DESKTOP"


def test_record_negative_metrics_rejected():
    with pytest.raises(ValueError):
        make_record(
            TrialSpec("text_entry", "baseline", 1), success=False,
            human_success=None, duration_s=-1, setup_duration_s=0,
            steps=0, failure_category="none", recovery_status="none",
            replan_calls=0, injected_count=0,
        )


def test_eligible_replan_fault_is_valid():
    trial = TrialSpec("text_entry", "enhanced", 2, "grounding_exhausted", "local-vlm")
    assert trial.fault == "grounding_exhausted"


def test_mode_comparison_only_pairs_same_task_and_fault():
    row = {
        "task": "browser_search", "fault": "none", "mode": "baseline",
        "replan_backend": "none", "runs": 10,
        "auto_success_rate": 0.5, "mean_duration_s": 7.0,
    }
    enhanced = {**row, "mode": "enhanced", "auto_success_rate": 0.8, "mean_duration_s": 8.5}
    different_fault = {**enhanced, "fault": "grounding_once"}
    comparisons = compare_modes([row, enhanced, different_fault])
    assert len(comparisons) == 1
    assert comparisons[0]["auto_success_gain_pp"] == 30.0
    assert comparisons[0]["mean_duration_delta_s"] == 1.5
