from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "screenagent"
OUTPUT_DIR = DATA_DIR / "splits"

SEED = 42
VAL_RATIO = 0.2

EXPECTED_COUNTS = {
    "normal": 3486,
    "negative": 1662,
}


def load_jsonl(path: Path) -> list[dict]:
    """Read JSONL and validate each record."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run scripts/export_screenagent.py first."
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                raise ValueError(
                    f"Empty line: {path}:{line_number}"
                )

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record: {path}:{line_number}"
                )

            session_id = record.get(
                "metadata", {}
            ).get("session_id")

            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError(
                    f"Missing session_id: {path}:{line_number}"
                )

            records.append(record)

    return records


def get_session_id(record: dict) -> str:
    return record["metadata"]["session_id"]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def main():
    # 1. Load both datasets.
    datasets = {
        name: load_jsonl(DATA_DIR / f"{name}.jsonl")
        for name in ("normal", "negative")
    }

    # 2. Validate the earlier audit counts.
    for name, expected in EXPECTED_COUNTS.items():
        actual = len(datasets[name])

        if actual != expected:
            raise ValueError(
                f"{name}: expected {expected} records, "
                f"found {actual}. Check the export."
            )

    # 3. Collect unique sessions from BOTH datasets.
    all_sessions = sorted({
        get_session_id(record)
        for records in datasets.values()
        for record in records
    })

    if len(all_sessions) < 2:
        raise ValueError(
            "At least two sessions are required."
        )

    # 4. Deterministic session-level split.
    rng = random.Random(SEED)
    rng.shuffle(all_sessions)

    val_count = max(
        1,
        min(
            len(all_sessions) - 1,
            round(len(all_sessions) * VAL_RATIO),
        ),
    )

    val_sessions = set(all_sessions[:val_count])
    train_sessions = set(all_sessions[val_count:])

    assert train_sessions.isdisjoint(val_sessions)

    # 5. Split both normal and negative with the SAME mapping.
    splits = {}
    stats = Counter()

    for name, records in datasets.items():
        train_records = []
        val_records = []

        for record in records:
            session_id = get_session_id(record)

            if session_id in val_sessions:
                val_records.append(record)
            else:
                train_records.append(record)

        splits[name] = {
            "train": train_records,
            "val": val_records,
        }

        stats[f"{name}_train"] = len(train_records)
        stats[f"{name}_val"] = len(val_records)

        # Check: no record is lost.
        assert (
            len(train_records) + len(val_records)
            == len(records)
        )

    # 6. Verify leakage across all four outputs.
    actual_train_sessions = {
        get_session_id(record)
        for group in splits.values()
        for record in group["train"]
    }

    actual_val_sessions = {
        get_session_id(record)
        for group in splits.values()
        for record in group["val"]
    }

    overlap = actual_train_sessions & actual_val_sessions

    if overlap:
        raise RuntimeError(
            f"Session leakage detected: {overlap}"
        )

    # 7. Write outputs only after validation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, group in splits.items():
        for split_name, records in group.items():
            path = OUTPUT_DIR / f"{name}_{split_name}.jsonl"

            write_jsonl(path, records)

    # 8. Save a reproducibility manifest.
    manifest = {
        "source": "screenagent",
        "source_split": "train",
        "seed": SEED,
        "strategy": "session_level",
        "validation_ratio": VAL_RATIO,
        "session_count": len(all_sessions),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "session_overlap": len(overlap),
        "record_counts": dict(stats),
    }

    manifest_path = OUTPUT_DIR / "split_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== ScreenAgent Dataset Split =====")

    print("Sessions:", len(all_sessions))
    print("Train sessions:", len(train_sessions))
    print("Val sessions:", len(val_sessions))

    print("\n[Record Counts]")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n[Validation]")
    print("Session overlap:", len(overlap))

    total = sum(stats.values())
    print("Total records:", total)

    if total != 5148:
        raise RuntimeError("Total record count mismatch.")

    print("\nSPLIT CHECK: PASS")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()