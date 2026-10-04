from __future__ import annotations

import argparse
from pathlib import Path

from gui_agent.datasets.webarena import load_webarena_collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test one task from WebArena test.raw.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to config_files/test.raw.json",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    tasks = load_webarena_collection(args.input)
    by_id = {task.task_id: task for task in tasks}

    if args.task_id not in by_id:
        raise SystemExit(f"task_id={args.task_id} not found")

    task = by_id[args.task_id]

    print("===== WebArena Real Data =====")
    print("Task ID:", task.task_id)
    print("Intent:", task.intent)
    print("Sites:", task.sites)
    print("Start URL:", task.start_url)
    print("Require login:", task.require_login)
    print("Require reset:", task.require_reset)
    print("Evaluation types:", task.eval_types)
    print("Reference action count:", task.reference_action_count)
    print("\nWEBARENA ADAPTER: PASS")


if __name__ == "__main__":
    main()
