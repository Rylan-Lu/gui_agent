import pytest

from gui_agent.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
)


def test_system_prompt_contains_json_format():
    assert '"steps"' in PLANNER_SYSTEM_PROMPT
    assert '"step_id"' in PLANNER_SYSTEM_PROMPT
    assert '"description"' in PLANNER_SYSTEM_PROMPT


def test_system_prompt_prohibits_execution():
    assert "Do not execute any actions" in PLANNER_SYSTEM_PROMPT


def test_build_planner_prompt():
    prompt = build_planner_prompt(
        "Open browser"
    )

    assert "Open browser" in prompt
    assert "Generate a high-level execution plan" in prompt


def test_instruction_whitespace():
    prompt = build_planner_prompt(
        "  Open browser  "
    )

    assert "Open browser" in prompt
    assert "  Open browser  " not in prompt


@pytest.mark.parametrize(
    "instruction",
    ["", "   ", None, 123],
)
def test_invalid_instruction(instruction):
    with pytest.raises(ValueError):
        build_planner_prompt(instruction)