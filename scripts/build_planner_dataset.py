from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "screenagent"
SPLIT_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "planner"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(
                    f"Empty line: {path}:{line_number}"
                )

            records.append(json.loads(line))

    return records


def build_planner_samples(records: list[dict]) -> list[dict]:
    """
    Convert action-level records into task-level
    planning examples.

    Grouping key: task_id (one source JSON file).
    """

    groups = defaultdict(list)

    for record in records:
        groups[record["task_id"]].append(record)

    samples = []

    for task_id, group in sorted(groups.items()):
        group.sort(key=lambda x: x["step_index"])

        first = group[0]

        # All actions from the same source file
        # must share the same task and screenshot.
        for record in group:
            if record["instruction"] != first["instruction"]:
                raise ValueError(
                    f"Instruction mismatch: {task_id}"
                )

            if record["screenshot"] != first["screenshot"]:
                raise ValueError(
                    f"Screenshot mismatch: {task_id}"
                )

        # Keep planning actions only.
        plan_steps = []

        for record in group:
            action = record.get("action")

            if not isinstance(action, dict):
                continue

            if action.get("action_type") != "plan":
                continue

            element = action.get("element")

            if isinstance(element, str) and element.strip():
                plan_steps.append(element.strip())

        # No planning label: do not invent one.
        if not plan_steps:
            continue

        sample = {
            "sample_id": task_id,
            "source": "screenagent",
            "session_id": first["metadata"]["session_id"],
            "instruction": first["instruction"],
            "screenshot": first["screenshot"],
            "plan_steps": plan_steps,
            "num_steps": len(plan_steps),
        }

        samples.append(sample)

    return samples


def write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )


def main():
    manifest_path = SPLIT_DIR / "split_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Missing split manifest. "
            "Run split_screenagent.py first."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("session_overlap") != 0:
        raise RuntimeError("Invalid source split.")

    results = {}
    session_sets = {}

    # Use NORMAL data only.
    for split in ("train", "val"):
        path = SPLIT_DIR / f"normal_{split}.jsonl"

        records = load_jsonl(path)

        expected = manifest["record_counts"][f"normal_{split}"]

        if len(records) != expected:
            raise RuntimeError(
                f"Source count mismatch: {split}"
            )

        samples = build_planner_samples(records)

        if not samples:
            raise RuntimeError(
                f"No planner samples found: {split}"
            )

        results[split] = samples

        session_sets[split] = {
            sample["session_id"]
            for sample in samples
        }

    # Planner-level leakage check.
    overlap = session_sets["train"] & session_sets["val"]

    if overlap:
        raise RuntimeError(
            f"Planner session leakage: {overlap}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, samples in results.items():
        output = OUTPUT_DIR / f"planner_{split}.jsonl"

        write_jsonl(output, samples)

        print(f"{split}_samples: {len(samples)}")

        print(
            f"{split}_steps: "
            f"{sum(x['num_steps'] for x in samples)}"
        )

    print("session_overlap:", len(overlap))
    print("PLANNER DATASET CHECK: PASS")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()