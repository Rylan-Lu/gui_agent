from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.mind2web_pipeline import (
    export_mind2web_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Mind2Web train shards to compact processed JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing train_*.json or one train_N.json shard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Processed output directory.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="Pinned Hugging Face dataset revision/SHA.",
    )

    args = parser.parse_args()

    manifest = export_mind2web_dataset(
        args.input,
        args.output,
        source_revision=args.source_revision,
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nOutput: {args.output}")
    print("MIND2WEB EXPORT: PASS")


if __name__ == "__main__":
    main()
