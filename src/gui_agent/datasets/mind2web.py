from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gui_agent.datasets.schema import (
    ActionType,
    GUIAction,
    GUIExample,
)


_TARGET_LABEL_FIELDS = (
    "aria_label",
    "aria-label",
    "title",
    "text",
    "value",
    "placeholder",
    "name",
)


def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _parse_bbox(value: Any) -> dict[str, float] | None:
    """Parse Mind2Web's serialized DOM bounding_box_rect.

    The bbox belongs to the web document / cleaned HTML representation. It is
    intentionally kept in metadata and must not be treated as a desktop pixel
    coordinate without a viewport/screenshot transform.
    """

    if value is None or value == "":
        return None

    if not isinstance(value, str):
        raise ValueError("bounding_box_rect must be a string when present")

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "bounding_box_rect must contain x,y,width,height"
        )

    numbers: list[float] = []
    for part in parts:
        try:
            number = float(part)
        except ValueError as exc:
            raise ValueError(
                "bounding_box_rect values must be numeric"
            ) from exc

        if not math.isfinite(number):
            raise ValueError(
                "bounding_box_rect values must be finite"
            )
        numbers.append(number)

    x, y, width, height = numbers
    if width < 0 or height < 0:
        raise ValueError(
            "bounding_box_rect width/height must be >= 0"
        )

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def normalize_mind2web_candidate(
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one Mind2Web element candidate without losing semantics."""

    if not isinstance(raw, dict):
        raise ValueError("Mind2Web candidate must be an object")

    tag = _require_nonempty_string(
        raw.get("tag"),
        field_name="candidate.tag",
    )
    backend_node_id = _require_nonempty_string(
        raw.get("backend_node_id"),
        field_name="candidate.backend_node_id",
    )

    is_original_target = raw.get("is_original_target")
    is_top_level_target = raw.get("is_top_level_target")

    if type(is_original_target) is not bool:
        raise ValueError(
            "candidate.is_original_target must be a boolean"
        )
    if type(is_top_level_target) is not bool:
        raise ValueError(
            "candidate.is_top_level_target must be a boolean"
        )

    raw_attributes = raw.get("attributes")
    if not isinstance(raw_attributes, str):
        raise ValueError("candidate.attributes must be a JSON string")

    try:
        attributes = json.loads(raw_attributes)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "candidate.attributes must contain valid JSON"
        ) from exc

    if not isinstance(attributes, dict):
        raise ValueError(
            "candidate.attributes JSON must decode to an object"
        )

    bbox = _parse_bbox(attributes.get("bounding_box_rect"))

    labels = {
        key: value
        for key in _TARGET_LABEL_FIELDS
        if (value := attributes.get(key)) is not None
    }

    return {
        "tag": tag,
        "backend_node_id": backend_node_id,
        "is_original_target": is_original_target,
        "is_top_level_target": is_top_level_target,
        "bbox": bbox,
        "labels": labels,
        "attributes": attributes,
    }


def _normalize_candidate_list(
    value: Any,
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(value):
        try:
            normalized.append(
                normalize_mind2web_candidate(candidate)
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field_name}[{index}]: {exc}"
            ) from exc

    return normalized


def _validate_negative_candidate_list_count(
    value: Any,
    *,
    field_name: str,
) -> int:
    """Validate negative candidates using the fields needed by Mind2Web.

    ``neg_candidates`` are not positive target annotations.  The official
    Mind2Web action-prediction loader consumes their ``backend_node_id`` (and
    optional ranking fields added later), but does not rely on
    ``is_original_target`` / ``is_top_level_target``.  Real training shards
    contain negative candidates whose target flags are not booleans, so
    reusing the stricter positive-candidate normalizer here would reject valid
    source data.

    We therefore validate the container, each candidate object, and the stable
    element identifier required to address the candidate, without inventing
    positive-target semantics for negatives.
    """

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    for index, candidate in enumerate(value):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"Invalid {field_name}[{index}]: candidate must be an object"
            )

        try:
            _require_nonempty_string(
                candidate.get("backend_node_id"),
                field_name=f"{field_name}[{index}].backend_node_id",
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid {field_name}[{index}]: {exc}"
            ) from exc

    return len(value)


def parse_mind2web_action(raw: dict[str, Any]) -> GUIAction:
    """Convert one Mind2Web action into the unified GUIAction schema.

    ``pos_candidates`` are preserved as the ground-truth candidate set. An
    empty positive-candidate list is valid in Mind2Web because preprocessing
    may remove the annotated element from ``cleaned_html``.
    """

    if not isinstance(raw, dict):
        raise ValueError("Mind2Web action must be an object")

    action_uid = _require_nonempty_string(
        raw.get("action_uid"),
        field_name="action_uid",
    )

    # Validate the source HTML fields, but do not duplicate the large strings
    # into compact processed JSONL. source_file + task_id + action_uid is the
    # stable reference back to the raw shard.
    _require_string(raw.get("raw_html"), field_name="raw_html")
    _require_string(raw.get("cleaned_html"), field_name="cleaned_html")

    operation = raw.get("operation")
    if not isinstance(operation, dict):
        raise ValueError("operation must be an object")

    op = _require_nonempty_string(
        operation.get("op"),
        field_name="operation.op",
    )
    original_op = _require_nonempty_string(
        operation.get("original_op"),
        field_name="operation.original_op",
    )
    value = _require_string(
        operation.get("value"),
        field_name="operation.value",
    )

    positive_candidates = _normalize_candidate_list(
        raw.get("pos_candidates"),
        field_name="pos_candidates",
    )
    negative_candidate_count = _validate_negative_candidate_list_count(
        raw.get("neg_candidates"),
        field_name="neg_candidates",
    )

    original_target_candidates = [
        candidate
        for candidate in positive_candidates
        if candidate["is_original_target"]
    ]
    top_level_target_candidates = [
        candidate
        for candidate in positive_candidates
        if candidate["is_top_level_target"]
    ]

    metadata = {
        "action_uid": action_uid,
        "original_op": original_op,
        "operation": {
            "op": op,
            "original_op": original_op,
            "value": value,
        },
        "positive_candidates": positive_candidates,
        "positive_candidate_count": len(positive_candidates),
        "negative_candidate_count": negative_candidate_count,
        "original_target_candidates": original_target_candidates,
        "original_target_count": len(original_target_candidates),
        "top_level_target_count": len(top_level_target_candidates),
        "candidate_status": (
            "positive_candidates_available"
            if positive_candidates
            else "no_positive_candidate_in_cleaned_html"
        ),
    }

    # Mind2Web element boxes are DOM/web coordinates. They are deliberately
    # not copied into GUIAction.position.
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

    raise ValueError(f"Unsupported Mind2Web operation: {op}")


def parse_mind2web_task(
    data: dict[str, Any],
    source_file: str | None = None,
) -> list[GUIExample]:
    """Convert one Mind2Web task into step-level GUIExamples."""

    if not isinstance(data, dict):
        raise ValueError("Mind2Web task must be an object")

    task_id = _require_nonempty_string(
        data.get("annotation_id"),
        field_name="annotation_id",
    )
    instruction = _require_nonempty_string(
        data.get("confirmed_task"),
        field_name="confirmed_task",
    )
    website = _require_nonempty_string(
        data.get("website"),
        field_name="website",
    )
    domain = _require_nonempty_string(
        data.get("domain"),
        field_name="domain",
    )
    subdomain = _require_nonempty_string(
        data.get("subdomain"),
        field_name="subdomain",
    )

    actions = data.get("actions")
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")

    action_reprs = data.get("action_reprs")
    if not isinstance(action_reprs, list):
        raise ValueError("action_reprs must be a list")
    if len(action_reprs) != len(actions):
        raise ValueError(
            "action_reprs length must match actions length"
        )
    if not all(isinstance(item, str) for item in action_reprs):
        raise ValueError("action_reprs items must be strings")

    examples: list[GUIExample] = []
    history: list[GUIAction] = []

    for index, raw in enumerate(actions):
        try:
            action = parse_mind2web_action(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Mind2Web action at index {index}: {exc}"
            ) from exc

        example = GUIExample(
            source="mind2web",
            task_id=task_id,
            instruction=instruction,
            step_index=index,
            screenshot=None,
            action=action,
            history=tuple(history),
            metadata={
                "website": website,
                "domain": domain,
                "subdomain": subdomain,
                "action_uid": action.metadata["action_uid"],
                "action_repr": action_reprs[index],
                "source_file": source_file,
            },
        )

        examples.append(example)
        history.append(action)

    return examples


def load_mind2web_file(path: str | Path) -> list[GUIExample]:
    """Read one Mind2Web shard and return all step-level examples."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of Mind2Web tasks")

    examples: list[GUIExample] = []
    for task_index, task in enumerate(data):
        try:
            examples.extend(
                parse_mind2web_task(
                    task,
                    source_file=path.name,
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid Mind2Web task at index {task_index}: {exc}"
            ) from exc

    return examples
