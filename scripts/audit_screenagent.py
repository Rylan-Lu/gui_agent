from collections import Counter
from pathlib import Path

from gui_agent.datasets.schema import ActionType
from gui_agent.datasets.screenagent import load_screenagent_file


# ScreenAgent 仓库位于 GUI Agent 项目旁边
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT.parent
    / "ScreenAgent_reference"
    / "data"
    / "ScreenAgent"
    / "train"
)


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    files = sorted(DATA_ROOT.rglob("*.json"))

    stats = Counter()
    action_types = Counter()
    errors = []

    for path in files:
        stats["total_files"] += 1

        try:
            examples = load_screenagent_file(path)

            stats["parsed_files"] += 1
            stats["total_examples"] += len(examples)

            if "neg_plan" in path.stem.lower():
                stats["negative_files"] += 1
                stats["negative_examples"] += len(examples)
            else:
                stats["normal_files"] += 1
                stats["normal_examples"] += len(examples)

            if not examples:
                stats["empty_files"] += 1
                continue

            # 每个 JSON 的动作序号必须连续
            for index, example in enumerate(examples):

                if example.step_index != index:
                    stats["step_index_errors"] += 1

                if len(example.history) != index:
                    stats["history_errors"] += 1

                if example.action is None:
                    stats["missing_actions"] += 1
                    continue

                action_types[example.action.action_type.value] += 1

                if example.action.action_type == ActionType.OTHER:
                    stats["unknown_actions"] += 1

            # 一个 JSON 对应一张当前截图
            screenshot = examples[0].screenshot

            if screenshot is None:
                stats["missing_image_paths"] += 1
            elif not Path(screenshot).is_file():
                stats["missing_image_files"] += 1

        except Exception as exc:
            stats["failed_files"] += 1

            errors.append(
                {
                    "file": str(path),
                    "error": repr(exc),
                }
            )

    print("\n========== ScreenAgent Audit ==========")

    print("Dataset:", DATA_ROOT)

    print("\n[Files]")
    for key in (
        "total_files",
        "parsed_files",
        "failed_files",
        "normal_files",
        "negative_files",
        "empty_files",
    ):
        print(f"{key}: {stats[key]}")

    print("\n[Examples]")
    for key in (
        "total_examples",
        "normal_examples",
        "negative_examples",
    ):
        print(f"{key}: {stats[key]}")

    print("\n[Action Distribution]")
    for name, count in action_types.most_common():
        print(f"{name}: {count}")

    print("\n[Quality Checks]")
    for key in (
        "unknown_actions",
        "missing_actions",
        "missing_image_paths",
        "missing_image_files",
        "step_index_errors",
        "history_errors",
    ):
        print(f"{key}: {stats[key]}")

    print("\n[Parse Errors]")

    for error in errors[:10]:
        print(error)

    if len(errors) > 10:
        print(f"... and {len(errors) - 10} more errors")

    print("\n========== Audit Finished ==========")


if __name__ == "__main__":
    main()