"""Append privacy-conscious REAL desktop task outcomes after a manual E2E run.

This tool does not launch a desktop task or infer human review.
Use --summary to compute agreement and per-mode descriptive statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ALLOWED_STATUS = ("yes", "no", "unknown")


def parse_yes_no(value: str) -> bool | None:
    if value not in _ALLOWED_STATUS:
        raise ValueError(f"expected one of {_ALLOWED_STATUS}")
    return {"yes": True, "no": False, "unknown": None}[value]


def append_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_real(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for item in records:
        groups[(item["mode"], item["task_id"])].append(item)
    rows: list[dict] = []
    for (mode, task_id), group in sorted(groups.items()):
        reviewed = [g for g in group if g["human_success"] is not None]
        times = [g["duration_s"] for g in group]
        rows.append({
            "task_id": task_id,
            "mode": mode,
            "runs": len(group),
            "auto_successes": sum(g["auto_success"] for g in group),
            "reviewed_runs": len(reviewed),
            "human_successes": sum(g["human_success"] for g in reviewed),
            "auto_human_agreements": sum(
                g["auto_success"] == g["human_success"] for g in reviewed
            ),
            "false_positive_judge": sum(
                g["auto_success"] and not g["human_success"] for g in reviewed
            ),
            "false_negative_judge": sum(
                not g["auto_success"] and g["human_success"] for g in reviewed
            ),
            "mean_duration_s": round(statistics.mean(times), 3),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("artifacts/robustness/week6/real_runs.jsonl"))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--task-id", help="Stable non-sensitive task identifier")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--mode", choices=("baseline", "enhanced"))
    parser.add_argument("--auto-success", choices=_ALLOWED_STATUS)
    parser.add_argument("--human-success", choices=_ALLOWED_STATUS, default="unknown")
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--failure-category", default="none")
    parser.add_argument("--recovery-status", default="none")
    args = parser.parse_args(argv)
    if args.summary:
        rows = summarize_real(load_results(args.file))
        if not rows:
            print("No real-desktop observations recorded.")
            return 0
        target = args.file.with_name("real_summary.csv")
        with target.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Real-desktop summary: {target}")
        print("Unreviewed trials are EXCLUDED from human/auto agreement.")
        return 0
    for name in ("task_id", "trial", "mode", "auto_success", "duration_s", "steps"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} required unless --summary")
    if args.trial < 1 or args.duration_s < 0 or args.steps < 0:
        parser.error("trial must be >=1; duration and steps must be nonnegative")
    if parse_yes_no(args.auto_success) is None:
        parser.error("--auto-success must be yes or no for a completed run")
    record = {
        "source": "real_desktop_user_recorded",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "trial": args.trial,
        "mode": args.mode,
        "auto_success": parse_yes_no(args.auto_success),
        "human_success": parse_yes_no(args.human_success),
        "duration_s": args.duration_s,
        "steps": args.steps,
        "failure_category": args.failure_category,
        "recovery_status": args.recovery_status,
    }
    append_result(args.file, record)
    print(f"Recorded real run: {args.task_id} / {args.mode} / trial {args.trial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
