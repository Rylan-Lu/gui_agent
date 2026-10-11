"""Summarize actual human-reviewed desktop trials; never mix fault types."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty values")
    data = sorted(values)
    position = (len(data) - 1) * pct
    lo = math.floor(position)
    hi = math.ceil(position)
    return data[lo] + (data[hi] - data[lo]) * (position - lo)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("source") not in ("real_desktop_controlled", "real_desktop_natural"):
            continue  # mock records must never enter the real report
        # Do not pool CPU OCR with prior GPU OCR trials.
        key = (row["task_base"], row["fault"], row["mode"],
               row.get("replan_backend", "none"), row.get("ocr_device", "gpu:0"))
        grouped[key].append(row)
    output = []
    for (task, fault, mode, backend, ocr_device), batch in sorted(grouped.items()):
        times = [float(r["duration_s"]) for r in batch]
        reviewed = [r for r in batch if r.get("human_success") is not None]
        agreement = sum(r["auto_success"] == r["human_success"] for r in reviewed)
        output.append({
            "task": task,
            "fault": fault,
            "mode": mode,
            "replan_backend": backend,
            "ocr_device": ocr_device,
            "runs": len(batch),
            "fully_reviewed": len(reviewed),
            "auto_successes": sum(bool(r["auto_success"]) for r in batch),
            "auto_success_rate": round(sum(bool(r["auto_success"]) for r in batch) / len(batch), 4),
            "human_successes": sum(bool(r["human_success"]) for r in reviewed),
            "human_success_rate_reviewed": round(sum(bool(r["human_success"]) for r in reviewed) / len(reviewed), 4) if reviewed else "",
            "judge_agreement_reviewed": round(agreement / len(reviewed), 4) if reviewed else "",
            "judge_false_positive": sum(r["auto_success"] and not r["human_success"] for r in reviewed),
            "judge_false_negative": sum(not r["auto_success"] and r["human_success"] for r in reviewed),
            "mean_duration_s": round(statistics.mean(times), 3),
            "p95_duration_s": round(percentile(times, .95), 3),
            "std_duration_s": round(statistics.pstdev(times), 3),
            "mean_steps": round(statistics.mean(r["steps"] for r in batch), 3),
            "mean_replan_calls": round(statistics.mean(r.get("replan_calls", 0) for r in batch), 3),
            "sufficient_repetitions": len(batch) >= 10,
        })
    return output


def compare_modes(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive deltas for exact task+fault pairing; never pool fault types."""
    baseline = {
        (g["task"], g["fault"], g.get("ocr_device", "gpu:0")): g
        for g in groups if g["mode"] == "baseline" and g["replan_backend"] == "none"
    }
    comparisons = []
    for enhanced in groups:
        if enhanced["mode"] != "enhanced":
            continue
        reference = baseline.get((
            enhanced["task"], enhanced["fault"], enhanced.get("ocr_device", "gpu:0")
        ))
        if reference is None:
            continue
        comparisons.append({
            "task": enhanced["task"],
            "fault": enhanced["fault"],
            "enhanced_replan_backend": enhanced["replan_backend"],
            "ocr_device": enhanced.get("ocr_device", "gpu:0"),
            "baseline_n": reference["runs"],
            "enhanced_n": enhanced["runs"],
            "auto_success_gain_pp": round(100 * (enhanced["auto_success_rate"] - reference["auto_success_rate"]), 2),
            "mean_duration_delta_s": round(enhanced["mean_duration_s"] - reference["mean_duration_s"], 3),
            "both_modes_n_ge_10": reference["runs"] >= 10 and enhanced["runs"] >= 10,
        })
    return comparisons


def save_report(input_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    summaries = summarize(load_jsonl(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return []
    with (output_dir / "real_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    comparisons = compare_modes(summaries)
    if comparisons:
        with (output_dir / "real_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    (output_dir / "real_summary.json").write_text(
        json.dumps({"data_type": "REAL_DESKTOP", "groups": summaries, "comparisons": comparisons,
                    "method_note": "Natural and injected faults are separate; unreviewed trials excluded from human agreement; n<10 is preliminary."},
                   ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/robustness/week6/real_runs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/robustness/week6"))
    args = parser.parse_args(argv)
    groups = save_report(args.input, args.out)
    for group in groups:
        print(f"{group['task']} fault={group['fault']} {group['mode']} n={group['runs']} auto={group['auto_success_rate']:.0%} reviewed={group['fully_reviewed']} mean={group['mean_duration_s']}s")
    print("REAL DESKTOP ONLY; no mock samples; no human outcome inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
