from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.screenagent_pipeline import build_screenagent_planner_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "screenagent_rebuild" / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ScreenAgent task-level planner samples.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_screenagent_planner_dataset(args.split_dir, output_dir=args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    output = args.output if args.output is not None else args.split_dir.parent / "planner"
    print(f"\nOutput: {output}")
    print("SCREENAGENT PLANNER DATASET: PASS")


if __name__ == "__main__":
    main()
