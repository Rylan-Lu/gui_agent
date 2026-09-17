from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from gui_agent.datasets.mind2web import load_mind2web_file


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT.parent / "Mind2Web_reference"

SOURCE = (
    DATA_ROOT
    / "data"
    / "train"
    / "train_10.json"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "mind2web"

OUTPUT_FILE = OUTPUT_DIR / "train_10.jsonl"
STATS_FILE = OUTPUT_DIR / "train_10_stats.json"


def serialize_example(example):
    """Convert GUIExample to portable JSON data."""

    record = asdict(example)

    # Remove machine-specific absolute paths.
    source_file = record["metadata"].get("source_file")

    if source_file:
        record["metadata"]["source_file"] = (
            Path(source_file)
            .resolve()
            .relative_to(DATA_ROOT.resolve())
            .as_posix()
        )

    return record


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    print("===== Mind2Web Export =====")

    # Load original tasks.
    with SOURCE.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("Expected a list of tasks")

    raw_actions = sum(
        len(task["actions"])
        for task in tasks
    )

    # Use our existing adapter.
    examples = load_mind2web_file(SOURCE)

    if len(examples) != raw_actions:
        raise RuntimeError(
            "Action count mismatch: "
            f"{raw_actions} vs {len(examples)}"
        )

    action_counts = Counter(
        example.action.action_type.value
        for example in examples
    )

    if action_counts["other"] != 0:
        raise RuntimeError(
            "Unknown actions found during conversion"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file first.
    temp_file = OUTPUT_FILE.with_suffix(".jsonl.tmp")

    try:
        with temp_file.open("w", encoding="utf-8") as f:
            for example in examples:
                record = serialize_example(example)

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )

        # Read back and validate.
        exported = []

        with temp_file.open("r", encoding="utf-8") as f:
            for line in f:
                exported.append(json.loads(line))

        if len(exported) != len(examples):
            raise RuntimeError("Export count mismatch")

        exported_actions = Counter(
            row["action"]["action_type"]
            for row in exported
        )

        if exported_actions != action_counts:
            raise RuntimeError("Action distribution mismatch")

        # Replace the destination only after validation.
        os.replace(temp_file, OUTPUT_FILE)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    stats = {
        "source": "mind2web",
        "source_file": "data/train/train_10.json",
        "total_tasks": len(tasks),
        "raw_actions": raw_actions,
        "exported_examples": len(exported),
        "action_distribution": dict(action_counts),
        "unknown_actions": action_counts["other"],
        "output_file": OUTPUT_FILE.name,
    }

    STATS_FILE.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTasks:", len(tasks))
    print("Raw actions:", raw_actions)
    print("Exported examples:", len(exported))

    print("\nAction distribution:")

    for name, count in action_counts.most_common():
        print(f"{name}: {count}")

    print("\nOutput:", OUTPUT_FILE)
    print("Statistics:", STATS_FILE)

    print("\nEXPORT CHECK: PASS")


if __name__ == "__main__":
    main()