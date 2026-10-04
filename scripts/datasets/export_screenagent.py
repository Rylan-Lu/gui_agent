from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.screenagent_pipeline import (
    SCREENAGENT_DEFAULT_REPOSITORY,
    export_screenagent_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "screenagent_rebuild"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw ScreenAgent actions to portable JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="ScreenAgent train directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Processed output directory")
    parser.add_argument("--source-revision", required=True, help="Exact upstream Git commit/revision")
    parser.add_argument("--source-repository", default=SCREENAGENT_DEFAULT_REPOSITORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_screenagent_dataset(
        args.input,
        args.output,
        source_revision=args.source_revision,
        source_repository=args.source_repository,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nOutput: {args.output}")
    print("SCREENAGENT EXPORT: PASS")


if __name__ == "__main__":
    main()
