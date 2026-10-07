import json
from pathlib import Path

import pytest

from gui_agent.datasets.action_sft import (
    build_screenagent_action_sft_dataset,
    compact_action,
    normalize_key,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Control_L", "ctrl"),
        ("Control_R", "ctrl"),
        ("Shift_L", "shift"),
        ("Return", "enter"),
        ("Escape", "esc"),
        ("A", "a"),
    ],
)
def test_normalize_key(raw, expected):
    assert normalize_key(raw) == expected


def test_compact_click_action():
    action = {
        "action_type": "click",
        "position": {
            "x": 487.0,
            "y": 193.0,
        },
        "mouse_button": "left",
        "element": None,
    }

    assert compact_action(action) == {
        "action_type": "click",
        "position": {
            "x": 487,
            "y": 193,
        },
    }


def test_compact_hotkey_normalizes_keys():
    action = {
        "action_type": "hotkey",
        "keys": [
            "Control_L",
            "Shift_L",
            "A",
        ],
    }

    assert compact_action(action) == {
        "action_type": "hotkey",
        "keys": [
            "ctrl",
            "shift",
            "a",
        ],
    }


def test_compact_key_press_requires_one_key():
    with pytest.raises(ValueError):
        compact_action({
            "action_type": "key_press",
            "keys": ["ctrl", "a"],
        })


def test_unsupported_target_action_rejected():
    with pytest.raises(ValueError):
        compact_action({
            "action_type": "plan",
            "element": "do something",
        })


def _record(
    *,
    session_id: str,
    task_id: str,
    step_index: int,
    screenshot: str,
    action: dict,
    history: list[dict] | None = None,
) -> dict:
    return {
        "source": "screenagent",
        "task_id": task_id,
        "instruction": "test instruction",
        "step_index": step_index,
        "screenshot": screenshot,
        "action": action,
        "history": history or [],
        "metadata": {
            "session_id": session_id,
            "video_width": 1024,
            "video_height": 768,
        },
    }


def _write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def test_build_action_sft_dataset_has_no_session_leakage(
    tmp_path,
):
    split_dir = tmp_path / "splits"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "action_sft"

    train_records = []

    for index in range(4):
        session_id = f"train_session_{index}"

        screenshot = (
            f"{session_id}/images/screen.jpg"
        )

        image_path = image_root / screenshot
        image_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        image_path.touch()

        train_records.append(
            _record(
                session_id=session_id,
                task_id=f"task_{index}",
                step_index=0,
                screenshot=screenshot,
                action={
                    "action_type": "click",
                    "position": {
                        "x": 100 + index,
                        "y": 200 + index,
                    },
                    "mouse_button": "left",
                },
            )
        )

    test_screenshot = (
        "test_session/images/screen.jpg"
    )

    test_image = image_root / test_screenshot
    test_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    test_image.touch()

    test_records = [
        _record(
            session_id="test_session",
            task_id="test_task",
            step_index=0,
            screenshot=test_screenshot,
            action={
                "action_type": "type_text",
                "text": "hello",
            },
        )
    ]

    _write_jsonl(
        split_dir / "normal_train.jsonl",
        train_records,
    )

    _write_jsonl(
        split_dir / "normal_val.jsonl",
        test_records,
    )

    manifest = (
        build_screenagent_action_sft_dataset(
            split_dir,
            output_dir=output_dir,
            image_root=image_root,
            seed=42,
            dev_ratio=0.25,
            max_history=4,
        )
    )

    train = _read_jsonl(
        output_dir / "train.jsonl"
    )
    dev = _read_jsonl(
        output_dir / "dev.jsonl"
    )
    test = _read_jsonl(
        output_dir / "test.jsonl"
    )

    train_sessions = {
        sample["session_id"]
        for sample in train
    }

    dev_sessions = {
        sample["session_id"]
        for sample in dev
    }

    test_sessions = {
        sample["session_id"]
        for sample in test
    }

    assert not (
        train_sessions
        & dev_sessions
    )

    assert not (
        train_sessions
        & test_sessions
    )

    assert not (
        dev_sessions
        & test_sessions
    )

    assert test_sessions == {
        "test_session"
    }

    assert (
        manifest["session_overlap"]
        == {
            "train_dev": 0,
            "train_test": 0,
            "dev_test": 0,
        }
    )


def test_history_is_capped_and_compacted(
    tmp_path,
):
    split_dir = tmp_path / "splits"
    image_root = tmp_path / "images"
    output_dir = tmp_path / "output"

    screenshot = (
        "session_a/images/screen.jpg"
    )

    image_path = image_root / screenshot
    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image_path.touch()

    history = [
        {
            "action_type": "click",
            "position": {
                "x": index,
                "y": index,
            },
            "mouse_button": "left",
        }
        for index in range(6)
    ]

    records = [
        _record(
            session_id="session_a",
            task_id="task_a",
            step_index=6,
            screenshot=screenshot,
            history=history,
            action={
                "action_type": "key_press",
                "keys": ["Return"],
            },
        ),
        _record(
            session_id="session_b",
            task_id="task_b",
            step_index=0,
            screenshot=screenshot,
            action={
                "action_type": "wait",
                "wait_seconds": 0.5,
            },
        ),
    ]

    # Need at least two source-train sessions.
    _write_jsonl(
        split_dir / "normal_train.jsonl",
        records,
    )

    _write_jsonl(
        split_dir / "normal_val.jsonl",
        [
            _record(
                session_id="session_test",
                task_id="task_test",
                step_index=0,
                screenshot=screenshot,
                action={
                    "action_type": "scroll",
                    "scroll_delta": -1,
                },
            )
        ],
    )

    build_screenagent_action_sft_dataset(
        split_dir,
        output_dir=output_dir,
        image_root=image_root,
        seed=42,
        dev_ratio=0.5,
        max_history=4,
    )

    all_samples = (
        _read_jsonl(
            output_dir / "train.jsonl"
        )
        + _read_jsonl(
            output_dir / "dev.jsonl"
        )
    )

    sample = next(
        item
        for item in all_samples
        if item["task_id"] == "task_a"
    )

    assert len(sample["history"]) == 4

    assert sample["target"] == {
        "action_type": "key_press",
        "keys": ["enter"],
    }
def test_history_click_without_position_is_tolerated():
    action = {
        "action_type": "click",
        "position": None,
        "mouse_button": "left",
    }

    assert compact_action(
        action,
        require_supported=False,
    ) == {
        "action_type": "click",
    }
def test_target_click_without_position_is_rejected():
    with pytest.raises(ValueError):
        compact_action({
            "action_type": "click",
            "position": None,
        })