from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
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

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "screenagent"


def serialize_example(example):
    """Convert GUIExample to a portable JSON record."""

    record = asdict(example)

    # 不导出机器特有的绝对路径
    if record["screenshot"] is not None:
        record["screenshot"] = str(
            Path(record["screenshot"]).relative_to(DATA_ROOT)
        ).replace("\\", "/")

    source_file = record["metadata"].get("source_file")

    if source_file is not None:
        record["metadata"]["source_file"] = str(
            Path(source_file).relative_to(DATA_ROOT)
        ).replace("\\", "/")

    return record


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    normal_path = OUTPUT_DIR / "normal.jsonl"
    negative_path = OUTPUT_DIR / "negative.jsonl"

    files = sorted(DATA_ROOT.rglob("*.json"))

    stats = Counter()

    with (
        normal_path.open("w", encoding="utf-8") as normal_file,
        negative_path.open("w", encoding="utf-8") as negative_file,
    ):
        for path in files:
            examples = load_screenagent_file(path)

            stats["total_files"] += 1

            if not examples:
                stats["empty_files"] += 1
                continue

            is_negative = examples[0].metadata["is_negative_plan"]

            output = negative_file if is_negative else normal_file

            for example in examples:
                record = serialize_example(example)

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )

                if is_negative:
                    stats["negative_examples"] += 1
                else:
                    stats["normal_examples"] += 1

    stats["total_examples"] = (
        stats["normal_examples"]
        + stats["negative_examples"]
    )

    # 重新读取输出，检查 JSONL 是否有效以及行数是否一致
    for key, path in (
        ("normal_examples", normal_path),
        ("negative_examples", negative_path),
    ):
        count = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                json.loads(line)
                count += 1

        if count != stats[key]:
            raise RuntimeError(
                f"Count mismatch: {path}, "
                f"expected {stats[key]}, got {count}"
            )

    print("\n===== ScreenAgent Export =====")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nOutput files:")
    print(normal_path)
    print(negative_path)

    print("\nEXPORT CHECK: PASS")


if __name__ == "__main__":
    main()