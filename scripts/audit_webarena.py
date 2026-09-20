from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from gui_agent.datasets.webarena import load_webarena_file


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT.parent / "WebArena_reference"

DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "webarena" / "audit.json"
)


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError("Input must be a JSON file")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(path.glob("*.json"))

    if not files:
        raise ValueError("No JSON task files found")

    return files


def audit(files: list[Path]) -> dict:
    sites = Counter()
    eval_types = Counter()

    seen_ids = {}
    records = []
    errors = []

    reference_actions = 0

    for path in files:
        try:
            task = load_webarena_file(path)

            types = task.evaluation.get("eval_types", [])

            if not isinstance(types, list):
                raise ValueError("eval_types must be a list")

            if any(not isinstance(x, str) for x in types):
                raise ValueError("Invalid evaluation type")

            if task.task_id in seen_ids:
                raise ValueError(
                    f"Duplicate task_id={task.task_id}; "
                    f"previous file: {seen_ids[task.task_id]}"
                )

            seen_ids[task.task_id] = path.name

            sites.update(task.sites)

            eval_types.update(types)

            reference_actions += task.reference_action_count

            records.append({
                "source_file": path.name,
                "task_id": task.task_id,
                "sites": list(task.sites),
                "require_login": task.require_login,
                "require_reset": task.require_reset,
                "eval_types": types,
                "reference_action_count":
                    task.reference_action_count,
            })

        except Exception as exc:
            errors.append({
                "source_file": path.name,
                "error": repr(exc),
            })

    return {
        "total_files": len(files),
        "parsed_tasks": len(records),
        "failed_files": len(errors),
        "site_distribution": dict(sites),
        "evaluation_distribution": dict(eval_types),
        "reference_actions": reference_actions,
        "records": records,
        "errors": errors,
        "status": "PASS" if not errors else "FAILED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit WebArena task configurations"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    files = discover_files(args.input)

    report = audit(files)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = args.output.with_suffix(".json.tmp")

    try:
        temp.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(temp, args.output)

    finally:
        if temp.exists():
            temp.unlink()

    print("===== WebArena Batch Audit =====")

    print("Total files:", report["total_files"])
    print("Parsed tasks:", report["parsed_tasks"])
    print("Failed files:", report["failed_files"])
    print(
        "Reference actions:",
        report["reference_actions"],
    )
    print("Sites:", report["site_distribution"])
    print("Evaluation:", report["evaluation_distribution"])

    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print(error)

    print("\nReport:", args.output)
    print("AUDIT:", report["status"])

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()