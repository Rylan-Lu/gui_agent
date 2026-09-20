from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WebArenaTask:
    """A normalized WebArena task configuration."""

    task_id: int
    intent: str
    sites: tuple[str, ...]
    start_url: str

    require_login: bool
    require_reset: bool

    evaluation: dict[str, Any]
    reference_action_sequence: dict[str, Any] | None

    source_file: str | None = None

    @property
    def reference_action_count(self) -> int:
        """Return the number of reference actions, if available."""

        if self.reference_action_sequence is None:
            return 0

        actions = self.reference_action_sequence.get(
            "action_sequence"
        )

        if not isinstance(actions, list):
            return 0

        return len(actions)


def parse_webarena_task(
    data: dict[str, Any],
    source_file: str | None = None,
) -> WebArenaTask:
    """Convert a WebArena JSON object into WebArenaTask."""

    if not isinstance(data, dict):
        raise ValueError("Task must be a JSON object")

    task_id = data.get("task_id")
    intent = data.get("intent")
    sites = data.get("sites")
    start_url = data.get("start_url")

    if type(task_id) is not int or task_id < 0:
        raise ValueError("Invalid task_id")

    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Missing intent")

    if (
        not isinstance(sites, list)
        or not sites
        or any(
            not isinstance(site, str) or not site.strip()
            for site in sites
        )
    ):
        raise ValueError("Invalid sites")

    if not isinstance(start_url, str) or not start_url.strip():
        raise ValueError("Missing start_url")

    evaluation = data.get("eval")

    if not isinstance(evaluation, dict):
        raise ValueError("Invalid evaluation configuration")

    reference = data.get("reference_action_sequence")

    if reference is not None and not isinstance(reference, dict):
        raise ValueError("Invalid reference_action_sequence")

    require_login = data.get("require_login", False)
    require_reset = data.get("require_reset", False)

    if not isinstance(require_login, bool):
        raise ValueError("Invalid require_login")

    if not isinstance(require_reset, bool):
        raise ValueError("Invalid require_reset")

    return WebArenaTask(
        task_id=task_id,
        intent=intent.strip(),
        sites=tuple(sites),
        start_url=start_url,
        require_login=require_login,
        require_reset=require_reset,
        evaluation=dict(evaluation),
        reference_action_sequence=(
            dict(reference)
            if reference is not None
            else None
        ),
        source_file=source_file,
    )


def load_webarena_file(
    path: str | Path,
) -> WebArenaTask:
    """Load one WebArena task configuration."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return parse_webarena_task(
        data,
        source_file=str(path),
    )