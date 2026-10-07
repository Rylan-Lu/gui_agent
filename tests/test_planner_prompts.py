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


def test_system_prompt_requires_literal_visible_click_target():
    assert '"element" MUST be literal text' in PLANNER_SYSTEM_PROMPT
    assert '"text in Notepad"' in PLANNER_SYSTEM_PROMPT
    assert '"search field"' in PLANNER_SYSTEM_PROMPT


def test_system_prompt_defines_click_selection_semantics():
    assert "A click does NOT select a text phrase or range" in PLANNER_SYSTEM_PROMPT
    assert "Never hide selection semantics" in PLANNER_SYSTEM_PROMPT


def test_system_prompt_defines_delete_semantics():
    assert "Backspace/Delete" in PLANNER_SYSTEM_PROMPT
    assert "previous primitive steps have explicitly created that selection" in PLANNER_SYSTEM_PROMPT


def test_system_prompt_requires_description_action_consistency():
    assert "description MUST accurately describe" in PLANNER_SYSTEM_PROMPT
    assert "Do not describe a click as selecting text" in PLANNER_SYSTEM_PROMPT


def test_system_prompt_contains_primitive_text_editing_example():
    assert '"keys": ["end"]' in PLANNER_SYSTEM_PROMPT
    assert '["ctrl", "shift", "left"]' in PLANNER_SYSTEM_PROMPT
    assert '"action_type": "type_text"' in PLANNER_SYSTEM_PROMPT


def test_build_planner_prompt():
    prompt = build_planner_prompt(
        "Open browser"
    )

    assert "Open browser" in prompt
    assert "Generate a directly executable primitive GUI plan" in prompt


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
