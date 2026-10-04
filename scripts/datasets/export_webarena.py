from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.webarena_pipeline import export_webarena_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export WebArena test.raw.json into a reproducible task-level "
            "processed dataset"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to WebArena config_files/test.raw.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Processed output directory",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="WebArena source repository commit/revision",
    )
    args = parser.parse_args()

    manifest = export_webarena_dataset(
        args.input,
        args.output,
        source_revision=args.source_revision,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nOutput: {args.output}")
    print("WEBARENA EXPORT: PASS")


if __name__ == "__main__":
    main()
