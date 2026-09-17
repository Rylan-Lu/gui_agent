import json
from collections import Counter
from pathlib import Path

from gui_agent.datasets.screenagent import load_screenagent_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT.parent
    / "ScreenAgent_reference"
    / "data"
    / "ScreenAgent"
    / "train"
)


def main():
    files = sorted(DATA_ROOT.rglob("*.json"))

    stats = Counter()
    empty_files = []
    mismatches = []

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_actions = data.get("actions")

        # 不要把缺失、null、空列表混为一谈
        if "actions" not in data:
            stats["missing_actions_field"] += 1
            empty_files.append((str(path), "missing field"))
            continue

        if raw_actions is None:
            stats["null_actions"] += 1
            empty_files.append((str(path), "null"))
            continue

        if not isinstance(raw_actions, list):
            stats["invalid_actions_type"] += 1
            empty_files.append((str(path), "not a list"))
            continue

        stats["raw_actions"] += len(raw_actions)

        if len(raw_actions) == 0:
            stats["empty_action_lists"] += 1
            empty_files.append((str(path), "empty list"))

        for action in raw_actions:
            if not isinstance(action, dict):
                stats["invalid_action_items"] += 1
            elif "action_type" not in action:
                stats["missing_action_types"] += 1

        examples = load_screenagent_file(path)

        stats["parsed_examples"] += len(examples)

        if len(examples) != len(raw_actions):
            mismatches.append(
                {
                    "file": str(path),
                    "raw": len(raw_actions),
                    "parsed": len(examples),
                }
            )

    print("\n===== DATA INTEGRITY =====")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n===== EMPTY / INVALID FILES =====")

    for path, reason in empty_files:
        print(f"{reason}: {path}")

    print("\n===== COUNT MISMATCHES =====")

    for item in mismatches[:20]:
        print(item)

    print(f"\nTotal mismatches: {len(mismatches)}")

    if (
        stats["raw_actions"] == stats["parsed_examples"]
        and stats["invalid_action_items"] == 0
        and stats["missing_action_types"] == 0
        and not mismatches
    ):
        print("\nACTION COUNT CHECK: PASS")
    else:
        print("\nACTION COUNT CHECK: NEEDS REVIEW")


if __name__ == "__main__":
    main()