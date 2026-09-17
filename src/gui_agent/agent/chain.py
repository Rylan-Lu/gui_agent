from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda

from gui_agent.agent.planner import Planner


def validate_input(data: dict) -> dict:
    """Validate and normalize planning input."""

    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")

    instruction = data.get("instruction")

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must not be empty")

    screenshot = data.get("screenshot_path")

    if isinstance(screenshot, str):
        screenshot = Path(screenshot)

    if screenshot is not None and not isinstance(screenshot, Path):
        raise TypeError("screenshot_path must be a Path or None")

    return {
        "instruction": instruction.strip(),
        "screenshot_path": screenshot,
    }


def build_planning_chain(planner: Planner):
    """Build a LangChain planning pipeline."""

    validate = RunnableLambda(validate_input)

    generate_plan = RunnableLambda(
        lambda data: planner.plan(
            instruction=data["instruction"],
            screenshot_path=data["screenshot_path"],
        )
    )

    return validate | generate_plan