from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gui_agent.evaluation.action_eval import (
    ACTION_POLICY_PROMPTS,
    build_action_policy_prompt,
    parse_json_object,
    score_prediction,
    select_samples,
    summarize_results,
    validate_action_prediction,
)
from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import (
    LocalVLMClient,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "screenagent"
    / "action_sft"
    / "test.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "week5"
    / "base_action_eval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-VL on the "
            "ScreenAgent Action SFT test set."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help=(
            "Optional PEFT LoRA adapter directory. "
            "When omitted, evaluate the base model."
        ),
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model-id",
        default=(
            "Qwen/Qwen3-VL-2B-Instruct"
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--prompt-version",
        choices=sorted(
            ACTION_POLICY_PROMPTS
        ),
        default="base",
        help=(
            "Action policy system prompt "
            "used for this evaluation."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    samples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at "
                    f"{path}:{line_number}"
                ) from exc

            if not isinstance(sample, dict):
                raise ValueError(
                    f"sample must be an object at "
                    f"{path}:{line_number}"
                )

            samples.append(sample)

    return samples


def main() -> None:
    args = parse_args()

    samples = load_jsonl(
        args.dataset
    )

    samples = select_samples(
        samples,
        limit=args.limit,
        seed=args.seed,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        args.output / "results.jsonl"
    )

    summary_path = (
        args.output / "summary.json"
    )

    system_prompt = (
        ACTION_POLICY_PROMPTS[
            args.prompt_version
        ]
    )

    client = LocalVLMClient(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        adapter_path=args.adapter_path,
    )

    results = []

    try:
        with results_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for index, sample in enumerate(
                samples,
                start=1,
            ):
                screenshot = (
                    args.image_root
                    / sample["screenshot"]
                )

                if not screenshot.is_file():
                    raise FileNotFoundError(
                        screenshot
                    )

                prompt = (
                    build_action_policy_prompt(
                        sample
                    )
                )

                request = ModelRequest(
                    prompt=prompt,
                    image_path=screenshot,
                    system_prompt=(
                        system_prompt
                    ),
                )

                start = time.perf_counter()

                response = client.generate(
                    request
                )

                latency = (
                    time.perf_counter()
                    - start
                )

                json_parsed = False
                schema_valid = False
                prediction = None
                parse_error = None

                try:
                    parsed = parse_json_object(
                        response.text
                    )

                    json_parsed = True

                    prediction = (
                        validate_action_prediction(
                            parsed,
                            screen_size=(
                                sample[
                                    "screen_size"
                                ]
                            ),
                        )
                    )

                    schema_valid = True

                except (
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                ) as exc:
                    parse_error = str(exc)

                score = score_prediction(
                    sample["target"],
                    prediction,
                )

                result = {
                    "sample_id": (
                        sample["sample_id"]
                    ),
                    "prompt_version": (
                        args.prompt_version
                    ),
                    "target": (
                        sample["target"]
                    ),
                    "prediction": prediction,
                    "raw_output": (
                        response.text
                    ),
                    "json_parsed": (
                        json_parsed
                    ),
                    "schema_valid": (
                        schema_valid
                    ),
                    "parse_error": (
                        parse_error
                    ),
                    "score": score,
                    "latency_seconds": (
                        latency
                    ),
                    "model_name": (
                        response.model_name
                    ),
                }

                results.append(result)

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                print(
                    f"[{index}/{len(samples)}] "
                    f"prompt="
                    f"{args.prompt_version} "
                    f"target="
                    f"{sample['target']['action_type']} "
                    f"json={json_parsed} "
                    f"schema={schema_valid} "
                    f"type_ok="
                    f"{score['action_type_correct']} "
                    f"latency="
                    f"{latency:.2f}s"
                )

    finally:
        client.unload()

    metrics = summarize_results(
        results
    )

    summary = {
        "model_id": args.model_id,
        "adapter_path": (
            str(args.adapter_path.resolve())
            if args.adapter_path is not None
            else None
        ),
        "prompt_version": (
            args.prompt_version
        ),
        "dataset": str(
            args.dataset
        ),
        "image_root": str(
            args.image_root
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "seed": args.seed,
        "limit": args.limit,
        "metrics": metrics,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")

    print(
        "\nACTION POLICY EVALUATION: PASS"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nResults: {results_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()