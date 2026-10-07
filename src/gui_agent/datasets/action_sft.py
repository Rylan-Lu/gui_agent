from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_SFT_SCHEMA_VERSION = 1

SUPPORTED_ACTION_TYPES = frozenset({
    "click",
    "double_click",
    "type_text",
    "key_press",
    "hotkey",
    "scroll",
    "wait",
})


_KEY_ALIASES = {
    "Control_L": "ctrl",
    "Control_R": "ctrl",
    "Ctrl": "ctrl",
    "CTRL": "ctrl",
    "Shift_L": "shift",
    "Shift_R": "shift",
    "Alt_L": "alt",
    "Alt_R": "alt",
    "Return": "enter",
    "Enter": "enter",
    "Escape": "esc",
    "Esc": "esc",
    "BackSpace": "backspace",
    "Delete": "delete",
    "Tab": "tab",
    "space": "space",
    "Space": "space",
    "Left": "left",
    "Right": "right",
    "Up": "up",
    "Down": "down",
    "Home": "home",
    "End": "end",
    "Page_Up": "pageup",
    "Page_Down": "pagedown",
}


def normalize_key(key: Any) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("keyboard key must be a non-empty string")

    value = key.strip()

    if value in _KEY_ALIASES:
        return _KEY_ALIASES[value]

    return value.lower()


def _normalize_number(value: Any, *, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")

    if number.is_integer():
        return int(number)

    return number


def _compact_point(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise ValueError("position must be an object")

    return {
        "x": _normalize_number(
            value.get("x"),
            field_name="position.x",
        ),
        "y": _normalize_number(
            value.get("y"),
            field_name="position.y",
        ),
    }


def compact_action(
    action: dict[str, Any],
    *,
    require_supported: bool = True,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("action must be an object")

    action_type = action.get("action_type")

    if not isinstance(action_type, str) or not action_type:
        raise ValueError("action_type must be a non-empty string")

    if require_supported and action_type not in SUPPORTED_ACTION_TYPES:
        raise ValueError(
            f"unsupported Action SFT action_type: {action_type}"
        )

    result: dict[str, Any] = {
        "action_type": action_type,
    }

    if action_type in {
        "click",
        "double_click",
        "move",
        "mouse_down",
        "mouse_up",
    }:
        position = action.get("position")

        if position is not None:
            result["position"] = _compact_point(
                position
            )
        elif require_supported:
            raise ValueError(
                f"{action_type} requires position"
            )

        mouse_button = action.get("mouse_button")

        if (
                isinstance(mouse_button, str)
                and mouse_button
                and mouse_button != "left"
        ):
            result["mouse_button"] = mouse_button


    elif action_type == "drag":

        position = action.get("position")

        end_position = action.get("end_position")

        if position is not None:

            result["position"] = _compact_point(

                position

            )

        elif require_supported:

            raise ValueError(

                "drag requires position"

            )

        if end_position is not None:

            result["end_position"] = _compact_point(

                end_position

            )

        elif require_supported:

            raise ValueError(

                "drag requires end_position"

            )

    elif action_type == "key_press":
        keys = action.get("keys")

        if isinstance(keys, list) and len(keys) == 1:
            result["keys"] = [
                normalize_key(keys[0])
            ]

        elif require_supported:
            raise ValueError(
                "key_press requires exactly one key"
            )

        elif isinstance(keys, list) and keys:
            # History is contextual rather than a strict
            # execution target. Preserve usable raw keys.
            result["keys"] = [
                normalize_key(key)
                for key in keys
            ]


    elif action_type == "type_text":

        text = action.get("text")

        if isinstance(text, str):

            result["text"] = text

        elif require_supported:

            raise ValueError(

                "type_text requires string text"

            )



    elif action_type == "hotkey":

        keys = action.get("keys")

        if isinstance(keys, list) and keys:

            result["keys"] = [

                normalize_key(key)

                for key in keys

            ]


        elif require_supported:

            raise ValueError(

                "hotkey requires at least one key"

            )


    elif action_type == "scroll":
        scroll_delta = action.get("scroll_delta")

        if type(scroll_delta) is not int:
            raise ValueError(
                "scroll requires integer scroll_delta"
            )

        result["scroll_delta"] = scroll_delta

    elif action_type == "wait":
        wait_seconds = _normalize_number(
            action.get("wait_seconds"),
            field_name="wait_seconds",
        )

        if wait_seconds < 0:
            raise ValueError(
                "wait_seconds must be non-negative"
            )

        result["wait_seconds"] = wait_seconds

    else:
        # This path is used only for history actions when
        # require_supported=False.
        position = action.get("position")
        end_position = action.get("end_position")
        text = action.get("text")
        keys = action.get("keys")
        scroll_delta = action.get("scroll_delta")
        wait_seconds = action.get("wait_seconds")

        if position is not None:
            result["position"] = _compact_point(position)

        if end_position is not None:
            result["end_position"] = _compact_point(
                end_position
            )

        if isinstance(text, str):
            result["text"] = text

        if isinstance(keys, list) and keys:
            result["keys"] = [
                normalize_key(key)
                for key in keys
            ]

        if type(scroll_delta) is int:
            result["scroll_delta"] = scroll_delta

        if isinstance(
            wait_seconds,
            (int, float),
        ) and not isinstance(wait_seconds, bool):
            result["wait_seconds"] = _normalize_number(
                wait_seconds,
                field_name="wait_seconds",
            )

    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"record must be an object at "
                    f"{path}:{line_number}"
                )

            records.append(record)

    return records


def _atomic_write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    temp.replace(path)


def _atomic_write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temp.replace(path)


def _session_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(
            "ScreenAgent record is missing metadata"
        )

    session_id = metadata.get("session_id")

    if (
        not isinstance(session_id, str)
        or not session_id.strip()
    ):
        raise ValueError(
            "ScreenAgent record has invalid session_id"
        )

    return session_id.strip()


def _screen_size(
    record: dict[str, Any],
) -> tuple[int, int]:
    metadata = record.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(
            "ScreenAgent record is missing metadata"
        )

    width = metadata.get("video_width")
    height = metadata.get("video_height")

    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ValueError(
            "ScreenAgent record has invalid video resolution"
        )

    return width, height


def _build_sample(
    record: dict[str, Any],
    *,
    image_root: Path | None,
    max_history: int,
) -> dict[str, Any] | None:
    action = record.get("action")

    if not isinstance(action, dict):
        return None

    action_type = action.get("action_type")

    if action_type not in SUPPORTED_ACTION_TYPES:
        return None

    source = record.get("source")
    task_id = record.get("task_id")
    instruction = record.get("instruction")
    screenshot = record.get("screenshot")
    step_index = record.get("step_index", 0)

    if source != "screenagent":
        raise ValueError(
            f"unexpected source: {source!r}"
        )

    if (
        not isinstance(task_id, str)
        or not task_id.strip()
    ):
        raise ValueError("invalid task_id")

    if (
        not isinstance(instruction, str)
        or not instruction.strip()
    ):
        raise ValueError("invalid instruction")

    if (
        not isinstance(screenshot, str)
        or not screenshot.strip()
    ):
        raise ValueError("invalid screenshot path")

    if type(step_index) is not int or step_index < 0:
        raise ValueError("invalid step_index")

    session_id = _session_id(record)
    width, height = _screen_size(record)

    if image_root is not None:
        image_path = image_root / screenshot

        if not image_path.is_file():
            raise FileNotFoundError(
                f"missing ScreenAgent screenshot: "
                f"{image_path}"
            )

    raw_history = record.get("history")

    if raw_history is None:
        raw_history = []

    if not isinstance(raw_history, list):
        raise ValueError("history must be a list")

    selected_history = (
        raw_history[-max_history:]
        if max_history > 0
        else []
    )

    history: list[dict[str, Any]] = []

    for history_action in selected_history:
        if not isinstance(history_action, dict):
            raise ValueError(
                "history action must be an object"
            )

        history.append(
            compact_action(
                history_action,
                require_supported=False,
            )
        )

    target = compact_action(
        action,
        require_supported=True,
    )

    return {
        "sample_id": (
            f"{task_id}:{step_index}"
        ),
        "source": "screenagent",
        "session_id": session_id,
        "task_id": task_id,
        "step_index": step_index,
        "instruction": instruction.strip(),
        "screenshot": screenshot,
        "screen_size": {
            "width": width,
            "height": height,
        },
        "history": history,
        "target": target,
    }


def _convert_records(
    records: list[dict[str, Any]],
    *,
    image_root: Path | None,
    max_history: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    seen_sample_ids: set[str] = set()

    for record in records:
        sample = _build_sample(
            record,
            image_root=image_root,
            max_history=max_history,
        )

        if sample is None:
            continue

        sample_id = sample["sample_id"]

        if sample_id in seen_sample_ids:
            raise RuntimeError(
                f"duplicate Action SFT sample_id: "
                f"{sample_id}"
            )

        seen_sample_ids.add(sample_id)
        samples.append(sample)

    return samples


def _split_sessions(
    session_ids: set[str],
    *,
    seed: int,
    dev_ratio: float,
) -> tuple[set[str], set[str]]:
    if not 0 < dev_ratio < 1:
        raise ValueError(
            "dev_ratio must be between 0 and 1"
        )

    ordered = sorted(session_ids)

    if len(ordered) < 2:
        raise ValueError(
            "at least two training sessions are required"
        )

    rng = random.Random(seed)
    rng.shuffle(ordered)

    dev_count = round(
        len(ordered) * dev_ratio
    )

    dev_count = max(
        1,
        min(
            dev_count,
            len(ordered) - 1,
        ),
    )

    dev_sessions = set(
        ordered[:dev_count]
    )

    train_sessions = set(
        ordered[dev_count:]
    )

    return train_sessions, dev_sessions


def _summarize(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    actions = Counter()
    resolutions = Counter()
    sessions = set()

    for sample in samples:
        actions[
            sample["target"]["action_type"]
        ] += 1

        size = sample["screen_size"]

        resolutions[
            f"{size['width']}x{size['height']}"
        ] += 1

        sessions.add(
            sample["session_id"]
        )

    return {
        "samples": len(samples),
        "sessions": len(sessions),
        "action_distribution": dict(
            sorted(actions.items())
        ),
        "resolution_distribution": dict(
            sorted(resolutions.items())
        ),
    }


def build_screenagent_action_sft_dataset(
    split_dir: str | Path,
    *,
    output_dir: str | Path,
    image_root: str | Path | None = None,
    seed: int = 42,
    dev_ratio: float = 0.1,
    max_history: int = 4,
) -> dict[str, Any]:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    if type(max_history) is not int or max_history < 0:
        raise ValueError(
            "max_history must be a non-negative integer"
        )

    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    resolved_image_root = (
        Path(image_root)
        if image_root is not None
        else None
    )

    if (
        resolved_image_root is not None
        and not resolved_image_root.is_dir()
    ):
        raise FileNotFoundError(
            resolved_image_root
        )

    source_train = _load_jsonl(
        split_dir / "normal_train.jsonl"
    )
    source_test = _load_jsonl(
        split_dir / "normal_val.jsonl"
    )

    train_pool = _convert_records(
        source_train,
        image_root=resolved_image_root,
        max_history=max_history,
    )

    frozen_test = _convert_records(
        source_test,
        image_root=resolved_image_root,
        max_history=max_history,
    )

    train_pool_sessions = {
        sample["session_id"]
        for sample in train_pool
    }

    frozen_test_sessions = {
        sample["session_id"]
        for sample in frozen_test
    }

    source_overlap = (
        train_pool_sessions
        & frozen_test_sessions
    )

    if source_overlap:
        raise RuntimeError(
            "source train/validation session leakage: "
            f"{sorted(source_overlap)}"
        )

    train_sessions, dev_sessions = (
        _split_sessions(
            train_pool_sessions,
            seed=seed,
            dev_ratio=dev_ratio,
        )
    )

    train_samples = [
        sample
        for sample in train_pool
        if sample["session_id"]
        in train_sessions
    ]

    dev_samples = [
        sample
        for sample in train_pool
        if sample["session_id"]
        in dev_sessions
    ]

    test_samples = frozen_test

    if not train_samples:
        raise RuntimeError(
            "Action SFT train split is empty"
        )

    if not dev_samples:
        raise RuntimeError(
            "Action SFT dev split is empty"
        )

    if not test_samples:
        raise RuntimeError(
            "Action SFT test split is empty"
        )

    train_ids = {
        sample["session_id"]
        for sample in train_samples
    }

    dev_ids = {
        sample["session_id"]
        for sample in dev_samples
    }

    test_ids = {
        sample["session_id"]
        for sample in test_samples
    }

    if train_ids & dev_ids:
        raise RuntimeError(
            "train/dev session leakage"
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "train/test session leakage"
        )

    if dev_ids & test_ids:
        raise RuntimeError(
            "dev/test session leakage"
        )

    files = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "test": "test.jsonl",
    }

    _atomic_write_jsonl(
        output_dir / files["train"],
        train_samples,
    )

    _atomic_write_jsonl(
        output_dir / files["dev"],
        dev_samples,
    )

    _atomic_write_jsonl(
        output_dir / files["test"],
        test_samples,
    )

    manifest = {
        "dataset": "screenagent_action_sft",
        "schema_version": (
            ACTION_SFT_SCHEMA_VERSION
        ),
        "source_dataset": "screenagent",
        "split_strategy": (
            "session_level_train_dev_"
            "with_frozen_source_validation_test"
        ),
        "seed": seed,
        "dev_ratio": dev_ratio,
        "max_history": max_history,
        "supported_action_types": sorted(
            SUPPORTED_ACTION_TYPES
        ),
        "image_validation": {
            "enabled": (
                resolved_image_root is not None
            ),
            "missing_images": 0,
        },
        "splits": {
            "train": _summarize(
                train_samples
            ),
            "dev": _summarize(
                dev_samples
            ),
            "test": _summarize(
                test_samples
            ),
        },
        "session_overlap": {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        },
        "files": files,
    }

    _atomic_write_json(
        output_dir / "manifest.json",
        manifest,
    )

    return manifest