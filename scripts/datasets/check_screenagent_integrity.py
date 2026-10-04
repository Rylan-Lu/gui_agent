from __future__ import annotations

import argparse
import json
from pathlib import Path

from gui_agent.datasets.screenagent_pipeline import audit_screenagent_source, validate_source_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw ScreenAgent source integrity.")
    parser.add_argument("--input", required=True, type=Path, help="ScreenAgent train directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_screenagent_source(args.input)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    validate_source_audit(audit)
    print("\nSCREENAGENT INTEGRITY CHECK: PASS")


if __name__ == "__main__":
    main()
