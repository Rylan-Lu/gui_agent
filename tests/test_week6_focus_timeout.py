"""Offline checks for bounded CPU-OCR diagnosis and pre-typing focus guard."""
from unittest.mock import Mock

import pytest

from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.replan import ReplanContext, ReplanError, Replanner
from gui_agent.datasets.schema import ActionType, GUIAction
from scripts.robustness import week6_real_e2e as e2e


def context():
    return ReplanContext(
        task="send a unique marker", completed_steps=(),
        failed_step=PlanStep(
            1, "Focus local message field",
            GUIAction(action_type=ActionType.CLICK, element="E2E_TARGET"),
        ),
        pending_steps=(PlanStep(
            2, "Enter unique test marker",
            GUIAction(action_type=ActionType.TYPE_TEXT, text="W6_TEST"),
        ),),
        failure_reason="grounding target not found", attempt=1,
        available_steps=4,
    )


def plan(*actions):
    return TaskPlan(tuple(
        PlanStep(i, f"action {i}", action)
        for i, action in enumerate(actions, start=1)
    ))


def test_focus_guard_rejects_type_text_first():
    planner = Mock()
    planner.plan.return_value = plan(
        GUIAction(action_type=ActionType.TYPE_TEXT, text="W6_TEST"),
        GUIAction(action_type=ActionType.CLICK, element="SEND"),
    )
    with pytest.raises(ReplanError, match="restoring input focus"):
        Replanner(planner, require_focus_after_failed_click=True).replan(context())


def test_focus_guard_is_opt_in_for_legacy_plans():
    planner = Mock()
    planner.plan.return_value = plan(
        GUIAction(action_type=ActionType.TYPE_TEXT, text="W6_TEST")
    )
    assert Replanner(planner).replan(context()) == planner.plan.return_value


def test_focus_guard_allows_non_typing_first_action():
    planner = Mock()
    planner.plan.return_value = plan(
        GUIAction(action_type=ActionType.KEY_PRESS, keys=("tab",)),
        GUIAction(action_type=ActionType.TYPE_TEXT, text="W6_TEST"),
    )
    assert Replanner(planner, require_focus_after_failed_click=True).replan(context()) == planner.plan.return_value


def test_replan_prompt_mentions_focus_requirement():
    from gui_agent.agent.replan import build_replan_instruction
    assert "BEFORE typing" in build_replan_instruction(context())


def test_safe_failure_code_focus():
    assert e2e.safe_failure_code(
        "grounding miss; replanning failed: ReplanError: "
        "replan starts typing without restoring input focus"
    ) == "replan_missing_focus"


def test_cli_budget_forwarded(monkeypatch, tmp_path):
    got = []
    monkeypatch.setattr(e2e, "run_trial", lambda *a, **kw: got.append(kw))
    assert e2e.main([
        "--task", "text_entry", "--mode", "enhanced", "--trial", "9",
        "--ocr-device", "cpu", "--timeout-s", "180",
        "--output", str(tmp_path), "--execute",
    ]) == 0
    assert got[0]["ocr_device"] == "cpu"
    assert got[0]["timeout_s"] == 180.0


@pytest.mark.parametrize("bad", ["0", "-1", "nan", "inf", "301"])
def test_cli_rejects_invalid_budget(monkeypatch, bad):
    with pytest.raises(SystemExit):
        e2e.main([
            "--task", "text_entry", "--mode", "enhanced", "--trial", "9",
            "--timeout-s", bad, "--execute",
        ])
