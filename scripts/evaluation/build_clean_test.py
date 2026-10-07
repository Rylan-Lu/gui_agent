from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "data/processed/screenagent/action_sft"
)

SOURCE = ROOT / "test.jsonl"
OUTPUT = ROOT / "test_clean.jsonl"


EXPOSED_SAMPLE_IDS = {
    (
        "931dee7c509b4c1d99ee2920ebf04704:"
        "2023-12-20_14-55-25-921491_translate:0"
    ),
    (
        "a4641827ffa64079a2146eecf453d704:"
        "2023-12-20_17-22-35-408777_translate:0"
    ),
    (
        "931dee7c509b4c1d99ee2920ebf04704:"
        "2023-12-20_14-33-25-330668_translate:1"
    ),
    (
        "05f7b82ed5d74dbe9c7888b427fcc571:"
        "2023-12-19_14-01-59-268818_translate:1"
    ),
    (
        "738ce163c4c24a4b98e81e903b3f8202:"
        "2023-12-20_19-55-05-373089_translate:0"
    ),
}


def main() -> None:
    samples = []

    with SOURCE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                samples.append(
                    json.loads(line)
                )

    exposed_rows = [
        sample
        for sample in samples
        if sample["sample_id"]
        in EXPOSED_SAMPLE_IDS
    ]

    if len(exposed_rows) != len(
        EXPOSED_SAMPLE_IDS
    ):
        found = {
            sample["sample_id"]
            for sample in exposed_rows
        }

        missing = (
            EXPOSED_SAMPLE_IDS
            - found
        )

        raise RuntimeError(
            f"missing exposed samples: {missing}"
        )

    session_ids = {
        sample["session_id"]
        for sample in exposed_rows
        if sample.get("session_id")
        is not None
    }

    if session_ids:
        clean_samples = [
            sample
            for sample in samples
            if sample.get("session_id")
            not in session_ids
        ]

        strategy = (
            "exclude_entire_exposed_sessions"
        )
    else:
        clean_samples = [
            sample
            for sample in samples
            if sample["sample_id"]
            not in EXPOSED_SAMPLE_IDS
        ]

        strategy = (
            "exclude_exact_exposed_samples"
        )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for sample in clean_samples:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "source samples:",
        len(samples),
    )

    print(
        "exposed samples:",
        len(exposed_rows),
    )

    print(
        "exposed sessions:",
        len(session_ids),
    )

    print(
        "clean samples:",
        len(clean_samples),
    )

    print(
        "strategy:",
        strategy,
    )

    print(
        "output:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()