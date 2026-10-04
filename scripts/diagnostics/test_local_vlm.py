from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import LocalVLMClient


DEFAULT_PROMPT = (
    "Describe the visible GUI. "
    "Identify the application, main interface elements, "
    "and any clearly readable text. "
    "Do not invent details."
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real local-VLM diagnostic on one screenshot."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        type=Path,
        help="Exact screenshot to use.",
    )
    source.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Dataset directory to search recursively; "
            "the first image in sorted order is used."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    image_path = _resolve_image(
        image=args.image,
        data_root=args.data_root,
    )

    print("Image:", image_path)

    client = LocalVLMClient(
        max_new_tokens=args.max_new_tokens,
    )

    request = ModelRequest(
        prompt=args.prompt,
        image_path=image_path,
    )

    start = perf_counter()
    response = client.generate(request)
    elapsed = perf_counter() - start

    print("\n===== VLM RESULT =====")
    print("Model:", response.model_name)
    print("Response:", response.text)
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
