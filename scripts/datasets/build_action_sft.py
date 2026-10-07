from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.action_sft import (
    build_screenagent_action_sft_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
    / "splits"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
    / "action_sft"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ScreenAgent step-level "
            "GUI Action SFT train/dev/test data."
        )
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help=(
            "Raw ScreenAgent train directory "
            "containing session folders."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-history",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = (
        build_screenagent_action_sft_dataset(
            args.split_dir,
            output_dir=args.output,
            image_root=args.image_root,
            seed=args.seed,
            dev_ratio=args.dev_ratio,
            max_history=args.max_history,
        )
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\nOutput: {args.output}")
    print(
        "SCREENAGENT ACTION SFT DATASET: PASS"
    )


if __name__ == "__main__":
    main()