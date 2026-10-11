"""Context regression tests; do not touch real desktop or GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "robustness"))

from week6_real_e2e import TrialSpec, build_fixture, build_runtime_task  # noqa: E402
from gui_agent.agent.plan_schema import PlanStep  # noqa: E402
from gui_agent.agent.replan import ReplanContext, build_replan_instruction  # noqa: E402
from gui_agent.datasets.schema import ActionType, GUIAction  # noqa: E402


@pytest.mark.parametrize("task", ["text_entry", "browser_search", "browser_open"])
def test_runtime_task_includes_unique_marker(tmp_path, task):
    spec = TrialSpec(task, "enhanced", 1)
    fixture = build_fixture(spec, tmp_path)
    prompt = build_runtime_task(spec, fixture)
    assert fixture.marker in prompt
    if task == "browser_search":
        assert fixture.query in prompt
    if task == "text_entry":
        assert "E2E_TARGET" in prompt
        assert "SEND" in prompt


def test_replan_preserves_pending_text_and_keyboard_actions():
    context = ReplanContext(
        task="Type W6_TEST_MARKER and send it",
        completed_steps=(),
        failed_step=PlanStep(
            step_id=1,
            description="Click E2E_TARGET",
            action=GUIAction(action_type=ActionType.CLICK, element="E2E_TARGET"),
        ),
        pending_steps=(
            PlanStep(
                step_id=2, description="Type marker",
                action=GUIAction(action_type=ActionType.TYPE_TEXT, text="W6_TEST_MARKER"),
            ),
            PlanStep(
                step_id=3, description="Submit",
                action=GUIAction(action_type=ActionType.KEY_PRESS, keys=("enter",)),
            ),
        ),
        failure_reason="OCR target not found",
        attempt=1,
        available_steps=4,
    )
    prompt = build_replan_instruction(context)
    assert '"text": "W6_TEST_MARKER"' in prompt
    assert '"keys": [' in prompt
    assert '"enter"' in prompt
