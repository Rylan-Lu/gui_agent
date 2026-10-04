from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_EVAL_KEYS = (
    "eval_types",
    "reference_answers",
    "reference_url",
    "program_html",
)

_OPTIONAL_EVAL_TEXT_FIELDS = (
    "string_note",
    "reference_answer_raw_annotation",
    "url_note",
    "annotation_note",
)


@dataclass(frozen=True)
class WebArenaTask:
    """Normalized WebArena task configuration.

    WebArena test.raw.json is task-level benchmark configuration data,
    not an offline GUI action trajectory. The nested evaluation payload is
    therefore preserved instead of being coerced into GUIAction.
    """

    task_id: int
    intent: str
    sites: tuple[str, ...]
    start_url: str

    require_login: bool
    require_reset: bool

    storage_state: str | None
    geolocation: dict[str, Any] | None

    intent_template: str
    intent_template_id: int
    instantiation_dict: dict[str, Any]

    evaluation: dict[str, Any]

    # Present in WebArena example/single-task configs, but absent from the
    # canonical 812-task test.raw.json snapshot used by this project.
    reference_action_sequence: dict[str, Any] | None = None

    # Rare top-level note field in the raw benchmark snapshot.
    string_note: str | None = None

    source_file: str | None = None

    @property
    def eval_types(self) -> tuple[str, ...]:
        value = self.evaluation.get("eval_types", [])
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str))

    @property
    def reference_action_count(self) -> int:
        """Return the number of optional reference actions, if available."""

        if self.reference_action_sequence is None:
            return 0

        actions = self.reference_action_sequence.get("action_sequence")
        if not isinstance(actions, list):
            return 0

        return len(actions)


def _validate_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_sites(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("sites must be a non-empty list")

    sites: list[str] = []
    for index, site in enumerate(value):
        if not isinstance(site, str) or not site.strip():
            raise ValueError(
                "sites items must be non-empty strings "
                f"(index {index})"
            )
        sites.append(site.strip())

    return tuple(sites)


def _validate_evaluation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("eval must be an object")

    missing = [key for key in _REQUIRED_EVAL_KEYS if key not in value]
    if missing:
        raise ValueError(
            "eval is missing required field(s): " + ", ".join(missing)
        )

    eval_types = value["eval_types"]
    if not isinstance(eval_types, list) or not eval_types:
        raise ValueError("eval.eval_types must be a non-empty list")

    for index, item in enumerate(eval_types):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "eval.eval_types items must be non-empty strings "
                f"(index {index})"
            )

    reference_answers = value["reference_answers"]
    if reference_answers is not None and not isinstance(
        reference_answers,
        dict,
    ):
        raise ValueError("eval.reference_answers must be an object or null")

    reference_url = value["reference_url"]
    if reference_url is not None and not isinstance(reference_url, str):
        raise ValueError("eval.reference_url must be a string or null")

    program_html = value["program_html"]
    if not isinstance(program_html, list):
        raise ValueError("eval.program_html must be a list")

    for index, item in enumerate(program_html):
        if not isinstance(item, dict):
            raise ValueError(
                "eval.program_html items must be objects "
                f"(index {index})"
            )

    for field_name in _OPTIONAL_EVAL_TEXT_FIELDS:
        if field_name not in value:
            continue
        item = value[field_name]
        if not isinstance(item, str):
            raise ValueError(f"eval.{field_name} must be a string")

    # Preserve evaluator configuration exactly enough for later benchmark use.
    return copy.deepcopy(value)


def _validate_optional_reference_actions(
    value: Any,
) -> dict[str, Any] | None:
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError("reference_action_sequence must be an object or null")

    sequence = value.get("action_sequence")
    if sequence is not None and not isinstance(sequence, list):
        raise ValueError(
            "reference_action_sequence.action_sequence must be a list"
        )

    return copy.deepcopy(value)


def parse_webarena_task(
    data: dict[str, Any],
    source_file: str | None = None,
) -> WebArenaTask:
    """Convert one WebArena task object into :class:`WebArenaTask`."""

    if not isinstance(data, dict):
        raise ValueError("Task must be a JSON object")

    task_id = data.get("task_id")
    if type(task_id) is not int or task_id < 0:
        raise ValueError("task_id must be a non-negative integer")

    intent = _validate_nonempty_string(
        data.get("intent"),
        field_name="intent",
    )

    sites = _validate_sites(data.get("sites"))

    start_url = _validate_nonempty_string(
        data.get("start_url"),
        field_name="start_url",
    )

    require_login = data.get("require_login")
    if not isinstance(require_login, bool):
        raise ValueError("require_login must be a boolean")

    require_reset = data.get("require_reset")
    if not isinstance(require_reset, bool):
        raise ValueError("require_reset must be a boolean")

    storage_state = data.get("storage_state")
    if storage_state is not None and not isinstance(storage_state, str):
        raise ValueError("storage_state must be a string or null")

    geolocation = data.get("geolocation")
    if geolocation is not None and not isinstance(geolocation, dict):
        raise ValueError("geolocation must be an object or null")

    intent_template = data.get("intent_template")
    if not isinstance(intent_template, str):
        raise ValueError("intent_template must be a string")

    intent_template_id = data.get("intent_template_id")
    if type(intent_template_id) is not int or intent_template_id < 0:
        raise ValueError("intent_template_id must be a non-negative integer")

    instantiation_dict = data.get("instantiation_dict")
    if not isinstance(instantiation_dict, dict):
        raise ValueError("instantiation_dict must be an object")

    evaluation = _validate_evaluation(data.get("eval"))

    reference_action_sequence = _validate_optional_reference_actions(
        data.get("reference_action_sequence")
    )

    string_note = data.get("string_note")
    if string_note is not None and not isinstance(string_note, str):
        raise ValueError("string_note must be a string when present")

    return WebArenaTask(
        task_id=task_id,
        intent=intent,
        sites=sites,
        start_url=start_url,
        require_login=require_login,
        require_reset=require_reset,
        storage_state=storage_state,
        geolocation=(
            copy.deepcopy(geolocation)
            if geolocation is not None
            else None
        ),
        intent_template=intent_template,
        intent_template_id=intent_template_id,
        instantiation_dict=copy.deepcopy(instantiation_dict),
        evaluation=evaluation,
        reference_action_sequence=reference_action_sequence,
        string_note=string_note,
        source_file=source_file,
    )


def load_webarena_file(path: str | Path) -> WebArenaTask:
    """Load one single-task WebArena JSON configuration."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        raise ValueError(
            "Expected a single WebArena task object, got a collection; "
            "use load_webarena_collection()"
        )

    return parse_webarena_task(
        data,
        source_file=str(path),
    )


def load_webarena_collection(path: str | Path) -> list[WebArenaTask]:
    """Load a WebArena task collection such as ``test.raw.json``."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("WebArena collection root must be a list")

    tasks: list[WebArenaTask] = []

    for index, raw_task in enumerate(data):
        try:
            task = parse_webarena_task(
                raw_task,
                source_file=str(path),
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid WebArena task at index {index}: {exc}"
            ) from exc

        tasks.append(task)

    return tasks
