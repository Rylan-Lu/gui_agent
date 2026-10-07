from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from statistics import mean, median
from typing import Any

from gui_agent.datasets.action_sft import (
    SUPPORTED_ACTION_TYPES,
    normalize_key,
)


POINTER_ACTION_TYPES = frozenset({
    "click",
    "double_click",
})


ACTION_POLICY_BASE_SYSTEM_PROMPT = """
You are a GUI action prediction assistant.

Given:
- a user task,
- the current GUI screenshot,
- optional previous GUI actions,

predict exactly ONE next GUI action.

Allowed action types:

- click
- double_click
- type_text
- key_press
- hotkey
- scroll
- wait

For click and double_click:
- output pixel coordinates in the current screenshot.
- use the screenshot dimensions supplied in the prompt.
- position must contain x and y.

For type_text:
- output the exact text to type.

For key_press:
- output exactly one key.

For hotkey:
- output all keys in order.

For scroll:
- output integer scroll_delta.

For wait:
- output non-negative wait_seconds.

Return only one JSON object.
Do not include Markdown fences.
Do not include explanations.
""".strip()


ACTION_POLICY_OPTIMIZED_SYSTEM_PROMPT = """
You are a GUI action prediction assistant.

Given:
- a user task,
- the current GUI screenshot,
- optional previous GUI actions,

predict exactly ONE next GUI action.

You MUST use exactly one of these seven JSON schemas.

Click:
{"action_type":"click","position":{"x":123,"y":456}}

Double click:
{"action_type":"double_click","position":{"x":123,"y":456}}

Type text:
{"action_type":"type_text","text":"example"}

Key press:
{"action_type":"key_press","keys":["enter"]}

Hotkey:
{"action_type":"hotkey","keys":["ctrl","s"]}

Scroll:
{"action_type":"scroll","scroll_delta":-10}

Wait:
{"action_type":"wait","wait_seconds":0.5}

STRICT FORMAT RULES:

- The field name MUST be "action_type".
- NEVER use a field named "action".
- NEVER use "pixel_coords".
- NEVER represent coordinates as an array such as [123,456].
- For click and double_click, "position" MUST be:
  {"x":123,"y":456}
- Do not wrap the action inside another object.
- Do not add fields that are not shown above.
- Return exactly ONE JSON object.
- Do not include Markdown fences.
- Do not include explanations.

ACTION SELECTION RULES:

- Use the user task, screenshot, and previous actions together.
- Previous actions describe what has already happened.
- Do not repeat an action that has already been completed unless the
  screenshot clearly requires it.
- If text has already been typed and the next step is to submit it,
  prefer the appropriate key_press such as Enter.
- Distinguish click from double_click carefully.
- Use the screenshot dimensions supplied in the prompt when
  predicting pixel coordinates.
""".strip()


ACTION_POLICY_COMPACT_SYSTEM_PROMPT = """
Predict exactly ONE next GUI action from the user task,
current screenshot, and previous actions.

Previous actions have already happened. Predict what should happen NEXT.

Return ONLY valid JSON using exactly one format:

{"action_type":"click","position":{"x":100,"y":200}}
{"action_type":"click","position":{"x":100,"y":200},"mouse_button":"right"}
{"action_type":"double_click","position":{"x":100,"y":200}}
{"action_type":"type_text","text":"hello"}
{"action_type":"key_press","keys":["enter"]}
{"action_type":"hotkey","keys":["ctrl","s"]}
{"action_type":"scroll","scroll_delta":-10}
{"action_type":"wait","wait_seconds":0.5}

Rules:
- Output one JSON object only.
- All JSON property names must use double quotes.
- For normal left click, omit "mouse_button".
- For right or middle click, include "mouse_button".
- Do not output Markdown or explanations.
- Do not repeat an action already completed in previous actions.
""".strip()


ACTION_POLICY_PROMPTS = {
    "base": ACTION_POLICY_BASE_SYSTEM_PROMPT,
    "optimized": ACTION_POLICY_OPTIMIZED_SYSTEM_PROMPT,
    "compact": ACTION_POLICY_COMPACT_SYSTEM_PROMPT,
}


# Backward compatibility.
ACTION_POLICY_SYSTEM_PROMPT = (
    ACTION_POLICY_BASE_SYSTEM_PROMPT
)


_ACTION_FIELDS = {
    "click": {
        "action_type",
        "position",
        "mouse_button",
    },
    "double_click": {
        "action_type",
        "position",
        "mouse_button",
    },
    "type_text": {
        "action_type",
        "text",
    },
    "key_press": {
        "action_type",
        "keys",
    },
    "hotkey": {
        "action_type",
        "keys",
    },
    "scroll": {
        "action_type",
        "scroll_delta",
    },
    "wait": {
        "action_type",
        "wait_seconds",
    },
}


VALID_MOUSE_BUTTONS = frozenset({
    "left",
    "right",
    "middle",
})


def build_action_policy_prompt(
    sample: dict[str, Any],
) -> str:
    instruction = sample.get("instruction")
    history = sample.get("history")
    screen_size = sample.get("screen_size")

    if (
        not isinstance(instruction, str)
        or not instruction.strip()
    ):
        raise ValueError(
            "sample has invalid instruction"
        )

    if not isinstance(history, list):
        raise ValueError(
            "sample history must be a list"
        )

    if not isinstance(screen_size, dict):
        raise ValueError(
            "sample screen_size must be an object"
        )

    width = screen_size.get("width")
    height = screen_size.get("height")

    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ValueError(
            "sample has invalid screen_size"
        )

    history_json = json.dumps(
        history,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        f"Task: {instruction.strip()}\n"
        f"Screenshot: {width}x{height}\n"
        f"Previous actions: {history_json}\n"
        "Next action:"
    )


def clean_model_text(
    text: str,
) -> str:
    if not isinstance(text, str):
        raise ValueError(
            "model response must be a string"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "model response is empty"
        )

    if (
        text.startswith("```")
        and text.endswith("```")
    ):
        lines = text.splitlines()

        if len(lines) >= 3:
            lines = lines[1:-1]
            text = "\n".join(
                lines
            ).strip()

    return text


def parse_json_object(
    text: str,
) -> dict[str, Any]:
    cleaned = clean_model_text(
        text
    )

    data = json.loads(
        cleaned
    )

    if not isinstance(data, dict):
        raise ValueError(
            "prediction must be a JSON object"
        )

    return data


def _normalize_number(
    value: Any,
    *,
    field_name: str,
) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
    ):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite"
        )

    if number.is_integer():
        return int(number)

    return number


def _validate_position(
    value: Any,
    *,
    width: int,
    height: int,
) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError(
            "position must be an object"
        )

    if set(value) != {"x", "y"}:
        raise ValueError(
            "position must contain exactly x and y"
        )

    x = _normalize_number(
        value["x"],
        field_name="position.x",
    )

    y = _normalize_number(
        value["y"],
        field_name="position.y",
    )

    if not 0 <= x < width:
        raise ValueError(
            "position.x is outside the screenshot"
        )

    if not 0 <= y < height:
        raise ValueError(
            "position.y is outside the screenshot"
        )

    return {
        "x": x,
        "y": y,
    }


def _validate_mouse_button(
    value: Any,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "mouse_button must be a string"
        )

    if value not in VALID_MOUSE_BUTTONS:
        raise ValueError(
            "mouse_button must be "
            "left, right, or middle"
        )

    return value


def validate_action_prediction(
    data: dict[str, Any],
    *,
    screen_size: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(
            "prediction must be an object"
        )

    action_type = data.get(
        "action_type"
    )

    if action_type not in SUPPORTED_ACTION_TYPES:
        raise ValueError(
            f"unsupported action_type: "
            f"{action_type!r}"
        )

    allowed_fields = (
        _ACTION_FIELDS[action_type]
    )

    unknown = (
        set(data)
        - allowed_fields
    )

    if unknown:
        raise ValueError(
            "unexpected action fields: "
            f"{sorted(unknown)}"
        )

    width = screen_size.get(
        "width"
    )

    height = screen_size.get(
        "height"
    )

    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ValueError(
            "invalid screen_size"
        )

    result: dict[str, Any] = {
        "action_type": action_type,
    }

    if action_type in POINTER_ACTION_TYPES:
        result["position"] = (
            _validate_position(
                data.get("position"),
                width=width,
                height=height,
            )
        )

        mouse_button = (
            _validate_mouse_button(
                data.get(
                    "mouse_button",
                    "left",
                )
            )
        )

        # Canonical format:
        # omit ordinary left click.
        if mouse_button != "left":
            result["mouse_button"] = (
                mouse_button
            )

    elif action_type == "type_text":
        text = data.get(
            "text"
        )

        if not isinstance(text, str):
            raise ValueError(
                "type_text requires string text"
            )

        result["text"] = text

    elif action_type == "key_press":
        keys = data.get(
            "keys"
        )

        if (
            not isinstance(keys, list)
            or len(keys) != 1
        ):
            raise ValueError(
                "key_press requires "
                "exactly one key"
            )

        result["keys"] = [
            normalize_key(
                keys[0]
            )
        ]

    elif action_type == "hotkey":
        keys = data.get(
            "keys"
        )

        if (
            not isinstance(keys, list)
            or not keys
        ):
            raise ValueError(
                "hotkey requires "
                "at least one key"
            )

        result["keys"] = [
            normalize_key(key)
            for key in keys
        ]

    elif action_type == "scroll":
        scroll_delta = data.get(
            "scroll_delta"
        )

        if type(scroll_delta) is not int:
            raise ValueError(
                "scroll_delta must be an integer"
            )

        result["scroll_delta"] = (
            scroll_delta
        )

    elif action_type == "wait":
        wait_seconds = (
            _normalize_number(
                data.get(
                    "wait_seconds"
                ),
                field_name=(
                    "wait_seconds"
                ),
            )
        )

        if wait_seconds < 0:
            raise ValueError(
                "wait_seconds must be "
                "non-negative"
            )

        result["wait_seconds"] = (
            wait_seconds
        )

    return result


def coordinate_error(
    target: dict[str, Any],
    prediction: dict[str, Any],
) -> float:
    tx = float(
        target["position"]["x"]
    )

    ty = float(
        target["position"]["y"]
    )

    px = float(
        prediction["position"]["x"]
    )

    py = float(
        prediction["position"]["y"]
    )

    return math.hypot(
        px - tx,
        py - ty,
    )


def score_prediction(
    target: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    target_type = target[
        "action_type"
    ]

    result = {
        "target_action_type": (
            target_type
        ),
        "action_type_correct": False,
        "argument_exact": False,
        "exact_action_correct": False,
        "coordinate_error": None,
        "mouse_button_correct": None,
        "success_at_20px": False,
        "success_at_50px": False,
    }

    if prediction is None:
        return result

    prediction_type = (
        prediction.get(
            "action_type"
        )
    )

    type_correct = (
        prediction_type
        == target_type
    )

    result[
        "action_type_correct"
    ] = type_correct

    if not type_correct:
        return result

    if target_type in POINTER_ACTION_TYPES:
        error = coordinate_error(
            target,
            prediction,
        )

        result[
            "coordinate_error"
        ] = error

        target_button = (
            target.get(
                "mouse_button",
                "left",
            )
        )

        prediction_button = (
            prediction.get(
                "mouse_button",
                "left",
            )
        )

        button_correct = (
            prediction_button
            == target_button
        )

        result[
            "mouse_button_correct"
        ] = button_correct

        position_exact = (
            prediction["position"]
            == target["position"]
        )

        exact = (
            position_exact
            and button_correct
        )

        result[
            "argument_exact"
        ] = exact

        result[
            "exact_action_correct"
        ] = exact

        result[
            "success_at_20px"
        ] = (
            error <= 20.0
            and button_correct
        )

        result[
            "success_at_50px"
        ] = (
            error <= 50.0
            and button_correct
        )

        return result

    if target_type == "type_text":
        exact = (
            prediction.get(
                "text"
            )
            == target.get(
                "text"
            )
        )

    elif target_type in {
        "key_press",
        "hotkey",
    }:
        exact = (
            prediction.get(
                "keys"
            )
            == target.get(
                "keys"
            )
        )

    elif target_type == "scroll":
        exact = (
            prediction.get(
                "scroll_delta"
            )
            == target.get(
                "scroll_delta"
            )
        )

    elif target_type == "wait":
        exact = math.isclose(
            float(
                prediction.get(
                    "wait_seconds"
                )
            ),
            float(
                target.get(
                    "wait_seconds"
                )
            ),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    else:
        exact = False

    result[
        "argument_exact"
    ] = exact

    result[
        "exact_action_correct"
    ] = exact

    result[
        "success_at_20px"
    ] = exact

    result[
        "success_at_50px"
    ] = exact

    return result


def select_samples(
    samples: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if (
        limit is None
        or limit >= len(samples)
    ):
        return list(samples)

    if limit <= 0:
        raise ValueError(
            "limit must be positive"
        )

    buckets: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for sample in samples:
        action_type = (
            sample[
                "target"
            ][
                "action_type"
            ]
        )

        buckets[
            action_type
        ].append(
            sample
        )

    rng = random.Random(
        seed
    )

    for bucket in (
        buckets.values()
    ):
        rng.shuffle(
            bucket
        )

    action_types = sorted(
        buckets
    )

    selected: list[
        dict[str, Any]
    ] = []

    index = 0

    while len(selected) < limit:
        added = False

        for action_type in action_types:
            bucket = buckets[
                action_type
            ]

            if index < len(bucket):
                selected.append(
                    bucket[index]
                )

                added = True

                if (
                    len(selected)
                    == limit
                ):
                    break

        if not added:
            break

        index += 1

    return selected


def percentile_95(
    values: list[float],
) -> float | None:
    if not values:
        return None

    ordered = sorted(
        values
    )

    index = (
        math.ceil(
            0.95
            * len(ordered)
        )
        - 1
    )

    index = max(
        0,
        min(
            index,
            len(ordered) - 1,
        ),
    )

    return ordered[
        index
    ]


def summarize_results(
    results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    total = len(
        results
    )

    if total == 0:
        raise ValueError(
            "no evaluation results"
        )

    scores = [
        result["score"]
        for result in results
    ]

    pointer_scores = [
        score
        for score in scores
        if score[
            "target_action_type"
        ]
        in POINTER_ACTION_TYPES
    ]

    coordinate_errors = [
        float(
            score[
                "coordinate_error"
            ]
        )
        for score in pointer_scores
        if score[
            "coordinate_error"
        ]
        is not None
    ]

    latencies = [
        float(
            result[
                "latency_seconds"
            ]
        )
        for result in results
    ]

    warm_latencies = (
        latencies[1:]
    )

    per_action: dict[
        str,
        dict[str, Any],
    ] = {}

    action_types = sorted({
        score[
            "target_action_type"
        ]
        for score in scores
    })

    for action_type in action_types:
        subset = [
            score
            for score in scores
            if score[
                "target_action_type"
            ]
            == action_type
        ]

        count = len(
            subset
        )

        per_action[
            action_type
        ] = {
            "count": count,
            "action_type_accuracy": (
                sum(
                    item[
                        "action_type_correct"
                    ]
                    for item in subset
                )
                / count
            ),
            "exact_action_accuracy": (
                sum(
                    item[
                        "exact_action_correct"
                    ]
                    for item in subset
                )
                / count
            ),
            "success_at_50px": (
                sum(
                    item[
                        "success_at_50px"
                    ]
                    for item in subset
                )
                / count
            ),
        }

    pointer_button_scores = [
        score
        for score in pointer_scores
        if score[
            "mouse_button_correct"
        ]
        is not None
    ]

    latency_summary = {
        "all_mean": mean(
            latencies
        ),
        "all_median": median(
            latencies
        ),
        "all_p95": percentile_95(
            latencies
        ),
        "cold_first_request": (
            latencies[0]
        ),
        "warm_mean": (
            mean(
                warm_latencies
            )
            if warm_latencies
            else None
        ),
        "warm_median": (
            median(
                warm_latencies
            )
            if warm_latencies
            else None
        ),
        "warm_p95": (
            percentile_95(
                warm_latencies
            )
            if warm_latencies
            else None
        ),
    }

    return {
        "samples": total,
        "json_parse_rate": (
            sum(
                result[
                    "json_parsed"
                ]
                for result in results
            )
            / total
        ),
        "schema_valid_rate": (
            sum(
                result[
                    "schema_valid"
                ]
                for result in results
            )
            / total
        ),
        "action_type_accuracy": (
            sum(
                score[
                    "action_type_correct"
                ]
                for score in scores
            )
            / total
        ),
        "exact_action_accuracy": (
            sum(
                score[
                    "exact_action_correct"
                ]
                for score in scores
            )
            / total
        ),
        "success_at_20px": (
            sum(
                score[
                    "success_at_20px"
                ]
                for score in scores
            )
            / total
        ),
        "success_at_50px": (
            sum(
                score[
                    "success_at_50px"
                ]
                for score in scores
            )
            / total
        ),
        "pointer": {
            "targets": len(
                pointer_scores
            ),
            "valid_same_type_predictions": (
                len(
                    coordinate_errors
                )
            ),
            "mean_pixel_error": (
                mean(
                    coordinate_errors
                )
                if coordinate_errors
                else None
            ),
            "median_pixel_error": (
                median(
                    coordinate_errors
                )
                if coordinate_errors
                else None
            ),
            "point_at_20px": (
                sum(
                    score[
                        "success_at_20px"
                    ]
                    for score
                    in pointer_scores
                )
                / len(
                    pointer_scores
                )
                if pointer_scores
                else None
            ),
            "point_at_50px": (
                sum(
                    score[
                        "success_at_50px"
                    ]
                    for score
                    in pointer_scores
                )
                / len(
                    pointer_scores
                )
                if pointer_scores
                else None
            ),
            "mouse_button_accuracy": (
                sum(
                    score[
                        "mouse_button_correct"
                    ]
                    for score
                    in pointer_button_scores
                )
                / len(
                    pointer_button_scores
                )
                if pointer_button_scores
                else None
            ),
        },
        "latency_seconds": (
            latency_summary
        ),
        "per_action": (
            per_action
        ),
    }