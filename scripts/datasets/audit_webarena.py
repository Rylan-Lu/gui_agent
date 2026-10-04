from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gui_agent.datasets.webarena_audit import (
    audit_webarena_files,
    discover_webarena_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit WebArena task configurations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to test.raw.json or a directory of task JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional audit JSON output path",
    )
    args = parser.parse_args()

    files = discover_webarena_files(args.input)
    report = audit_webarena_files(files)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")

        try:
            temp.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, args.output)
        finally:
            if temp.exists():
                temp.unlink()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWEBARENA AUDIT: {report['status']}")

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
