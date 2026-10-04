from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from time import perf_counter

import torch

from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import LocalVLMClient


DEFAULT_PROMPT = "Briefly describe the visible GUI."


def _find_first_image(data_root: Path) -> Path:
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    images = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in data_root.rglob(pattern)
    )

    if not images:
        raise FileNotFoundError(
            f"No screenshots found under {data_root}"
        )

    return images[0]


def _resolve_image(
    *,
    image: Path | None,
    data_root: Path | None,
) -> Path:
    if image is not None:
        image = image.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        return image

    if data_root is None:
        raise ValueError("Either --image or --data-root is required")

    return _find_first_image(data_root.expanduser().resolve())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one local VLM screenshot request."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        type=Path,
        help="Exact screenshot used for all runs.",
    )
    source.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Dataset directory to search recursively; "
            "the first image in sorted order is used."
        ),
    )

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    image_path = _resolve_image(
        image=args.image,
        data_root=args.data_root,
    )

    client = LocalVLMClient(
        max_new_tokens=args.max_new_tokens,
    )
    request = ModelRequest(
        prompt=args.prompt,
        image_path=image_path,
    )

    print("===== LOCAL VLM BENCHMARK =====")
    print("Image:", image_path)
    print("Model:", client.model_id)
    print("Runs:", args.runs)
    print("Warmup:", args.warmup)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = perf_counter()
    client.load()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_time = perf_counter() - start
    print(f"Model load: {load_time:.3f}s")

    for index in range(args.warmup):
        client.generate(request)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Warmup {index + 1}: complete")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
    first_response_text: str | None = None

    for index in range(args.runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = perf_counter()
        response = client.generate(request)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = perf_counter() - start
        timings.append(elapsed)

        if first_response_text is None:
            first_response_text = response.text

        print(f"Inference {index + 1}: {elapsed:.3f}s")

    mean = statistics.mean(timings)
    p50 = _percentile(timings, 0.50)
    p95 = _percentile(timings, 0.95)
    std = statistics.pstdev(timings)

    print("\n===== SUMMARY =====")
    print("Raw timings (s):", [round(value, 4) for value in timings])
    print(f"Mean: {mean:.3f}s")
    print(f"P50:  {p50:.3f}s")
    print(f"P95:  {p95:.3f}s")
    print(f"Std:  {std:.3f}s")

    if torch.cuda.is_available():
        print(
            "Peak allocated GPU memory:",
            round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "GiB",
        )

    if first_response_text is not None:
        print("\nFirst measured response:")
        print(first_response_text)


if __name__ == "__main__":
    main()
