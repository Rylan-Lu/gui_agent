from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.dataset_summary import write_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified summary for processed GUI-agent datasets."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data") / "processed",
        help="Directory containing screenagent/, mind2web/, and webarena/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <processed-root>/dataset_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_dataset_summary(args.processed_root, args.output)
    output = args.output or args.processed_root / "dataset_summary.json"

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {output}")
    print("DATASET SUMMARY: PASS")


if __name__ == "__main__":
    main()
