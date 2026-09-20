from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    """One high-level step in a task plan."""

    step_id: int
    description: str

    def __post_init__(self) -> None:

        if type(self.step_id) is not int or self.step_id < 1:
            raise ValueError(
                "step_id must be a positive integer"
            )

        if (
            not isinstance(self.description, str)
            or not self.description.strip()
        ):
            raise ValueError(
                "description must not be empty"
            )


@dataclass(frozen=True)
class TaskPlan:
    """Validated high-level task plan."""

    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:

        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError(
                "steps must be a non-empty tuple"
            )

        for expected_id, step in enumerate(
            self.steps,
            start=1,
        ):
            if not isinstance(step, PlanStep):
                raise ValueError(
                    "Every step must be a PlanStep"
                )

            if step.step_id != expected_id:
                raise ValueError(
                    "step_id must be consecutive "
                    "and start from 1"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPlan:
        """Build a validated plan from a dictionary."""

        if not isinstance(data, dict):
            raise ValueError(
                "Plan must be a JSON object"
            )

        raw_steps = data.get("steps")

        if not isinstance(raw_steps, list):
            raise ValueError(
                "steps must be a list"
            )

        steps = []

        for item in raw_steps:

            if not isinstance(item, dict):
                raise ValueError(
                    "Each step must be an object"
                )

            steps.append(
                PlanStep(
                    step_id=item.get("step_id"),
                    description=item.get("description"),
                )
            )

        return cls(steps=tuple(steps))

    @classmethod
    def from_json(cls, text: str) -> TaskPlan:
        """Parse a model's JSON response."""

        if not isinstance(text, str):
            raise ValueError(
                "Model response must be a string"
            )

        data = json.loads(text)

        return cls.from_dict(data)