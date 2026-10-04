from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.screenagent_pipeline import split_screenagent_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "screenagent_rebuild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic session-level ScreenAgent split.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = split_screenagent_dataset(
        args.data_dir,
        output_dir=args.output,
        seed=args.seed,
        validation_ratio=args.val_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    output = args.output if args.output is not None else args.data_dir / "splits"
    print(f"\nOutput: {output}")
    print("SCREENAGENT SPLIT: PASS")


if __name__ == "__main__":
    main()
