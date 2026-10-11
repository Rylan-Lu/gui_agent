from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "robustness"))

from week6_fault_suite import (
    EXPECTED_SUCCESS,
    MODES,
    SCENARIOS,
    main,
    run_trial,
    summarize,
    write_reports,
)
from week6_real_record import (
    append_result,
    load_results,
    parse_yes_no,
    summarize_real,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", MODES)
def test_fault_injection_preserves_expected_outcome_and_safety(scenario, mode):
    trial = run_trial(scenario, mode)
    assert trial["auto_success"] == EXPECTED_SUCCESS[scenario][mode == "enhanced"]
    assert trial["safety_pass"], trial
    assert trial["dataset"] == "synthetic_fault_injection"


def test_two_mode_report_written_and_explicitly_labeled_simulation(tmp_path):
    records = [run_trial("delayed_grounding", mode, i) for i in (1, 2) for mode in MODES]
    stats = write_reports(records, tmp_path)
    assert len(stats) == 2
    assert (tmp_path / "synthetic_raw.csv").exists()
    assert (tmp_path / "synthetic_summary.csv").exists()
    info = json.loads((tmp_path / "synthetic_report.json").read_text(encoding="utf-8"))
    assert info["kind"] == "simulation_not_real_desktop"
    assert info["all_safety_checks_passed"] is True
    with (tmp_path / "synthetic_raw.csv").open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 4


def test_cli_returns_success_for_safe_matrix(tmp_path):
    code = main(["--runs", "1", "--out-dir", str(tmp_path)])
    assert code == 0
    assert len(summarize([run_trial("dispatch_unresolved", "enhanced")])) == 1


def test_real_review_summary_does_not_count_unknown_as_human_failure():
    records = [
        dict(task_id="notepad", mode="enhanced", auto_success=True,
             human_success=True, duration_s=2.0),
        dict(task_id="notepad", mode="enhanced", auto_success=True,
             human_success=None, duration_s=3.0),
        dict(task_id="notepad", mode="enhanced", auto_success=False,
             human_success=True, duration_s=4.0),
    ]
    row = summarize_real(records)[0]
    assert row["runs"] == 3
    assert row["reviewed_runs"] == 2
    assert row["human_successes"] == 2
    assert row["false_negative_judge"] == 1
    assert row["auto_human_agreements"] == 1


def test_real_record_append_and_read(tmp_path):
    path = tmp_path / "real.jsonl"
    append_result(path, {"task_id": "notepad", "auto_success": True})
    assert load_results(path) == [{"task_id": "notepad", "auto_success": True}]
    assert parse_yes_no("unknown") is None
    assert parse_yes_no("no") is False
