from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
    GUIExample,
)


def parse_mind2web_action(raw: dict[str, Any]) -> GUIAction:
    """Convert one Mind2Web action into GUIAction."""

    if not isinstance(raw, dict):
        raise ValueError("action must be a dictionary")

    operation = raw.get("operation")

    if not isinstance(operation, dict):
        raise ValueError("Missing operation")

    op = operation.get("op")
    value = operation.get("value")

    metadata = {
        "original_op": operation.get("original_op"),
        "operation": dict(operation),
        "action_uid": raw.get("action_uid"),
        "pos_candidates": raw.get("pos_candidates", []),
    }

    if op == "CLICK":
        return GUIAction(
            action_type=ActionType.CLICK,
            raw_action="CLICK",
            metadata=metadata,
        )

    if op == "TYPE":
        return GUIAction(
            action_type=ActionType.TYPE_TEXT,
            text=value,
            raw_action="TYPE",
            metadata=metadata,
        )

    if op == "SELECT":
        return GUIAction(
            action_type=ActionType.SELECT,
            text=value,
            raw_action="SELECT",
            metadata=metadata,
        )

    raise ValueError(
        f"Unsupported Mind2Web operation: {op}"
    )


def parse_mind2web_task(
    data: dict[str, Any],
    source_file: str | None = None,
) -> list[GUIExample]:
    """Convert one Mind2Web task into step-level examples."""

    task_id = data.get("annotation_id")
    instruction = data.get("confirmed_task")
    actions = data.get("actions")

    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Missing annotation_id")

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing confirmed_task")

    if not isinstance(actions, list):
        raise ValueError("actions must be a list")

    examples = []
    history = []

    for index, raw in enumerate(actions):

        action = parse_mind2web_action(raw)

        example = GUIExample(
            source="mind2web",
            task_id=task_id,
            instruction=instruction,
            step_index=index,

            # Standard Mind2Web data does not
            # guarantee a screenshot.
            screenshot=None,

            action=action,
            history=tuple(history),

            metadata={
                "website": data.get("website"),
                "domain": data.get("domain"),
                "action_uid": raw.get("action_uid"),
                "source_file": source_file,
            },
        )

        examples.append(example)
        history.append(action)

    return examples


def load_mind2web_file(path: str | Path) -> list[GUIExample]:
    """Read a JSON file containing Mind2Web tasks."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of Mind2Web tasks"
        )

    examples = []

    for task in data:
        examples.extend(
            parse_mind2web_task(
                task,
                source_file=str(path),
            )
        )

    return examples