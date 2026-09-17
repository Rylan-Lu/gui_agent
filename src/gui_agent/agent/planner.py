from __future__ import annotations

import re
from pathlib import Path

from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
)
from gui_agent.models.base import (
    ModelClient,
    ModelRequest,
)


class PlannerOutputError(ValueError):
    """Raised when a model returns an invalid task plan."""


class Planner:
    """Generate validated high-level GUI task plans."""

    def __init__(self, model: ModelClient):
        self.model = model

    @staticmethod
    def _clean_response(text: str) -> str:
        """Remove an optional JSON Markdown fence."""

        if not isinstance(text, str) or not text.strip():
            raise PlannerOutputError("Model returned an empty response")

        text = text.strip()

        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match:
            text = match.group(1).strip()

        return text

    def plan(
        self,
        instruction: str,
        screenshot_path: Path | None = None,
    ) -> TaskPlan:
        """Generate and validate a task plan."""

        # 1. Build prompt.
        prompt = build_planner_prompt(instruction)

        # 2. Construct model request.
        request = ModelRequest(
            prompt=prompt,
            image_path=screenshot_path,
            system_prompt=PLANNER_SYSTEM_PROMPT,
        )

        # 3. Call the model.
        response = self.model.generate(request)

        # 4. Normalize response formatting.
        text = self._clean_response(response.text)

        # 5. Parse and validate JSON.
        try:
            return TaskPlan.from_json(text)

        except (ValueError, TypeError) as exc:
            raise PlannerOutputError(
                "Model returned an invalid task plan: "
                f"{exc}"
            ) from exc