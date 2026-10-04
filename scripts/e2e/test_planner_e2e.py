from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from gui_agent.agent.planner import (
    Planner,
    PlannerOutputError,
)
from gui_agent.models.base import (
    ModelRequest,
    ModelResponse,
)
from gui_agent.models.local_vlm import LocalVLMClient


DEFAULT_INSTRUCTION = (
    "In the currently open Firefox browser, "
    "search for von Neumann and review the search results."
)


class RecordingModel:
    """Record the raw model response for debugging."""

    def __init__(self, model):
        self.model = model
        self.last_response: ModelResponse | None = None

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        response = self.model.generate(request)
        self.last_response = response
        return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real local-VLM -> Planner -> TaskPlan path "
            "against one screenshot."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Screenshot used by the Planner.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    print("===== REAL PLANNER TEST =====")
    print("Screenshot:", image_path)
    print("\nTask:", args.instruction)

    local_model = LocalVLMClient(
        max_new_tokens=args.max_new_tokens,
    )
    recording_model = RecordingModel(local_model)
    planner = Planner(model=recording_model)

    start = perf_counter()

    try:
        plan = planner.plan(
            instruction=args.instruction,
            screenshot_path=image_path,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = perf_counter() - start

        print("\n===== RAW MODEL OUTPUT =====")
        if recording_model.last_response is None:
            raise RuntimeError("Planner returned without a model response")
        print(recording_model.last_response.text)

        print("\n===== VALIDATED TASK PLAN =====")
        for step in plan.steps:
            print(f"{step.step_id}. {step.description}")

        print("\nStep count:", len(plan.steps))
        print(f"Total time: {elapsed:.2f}s")
        print("\nPLANNER E2E: PASS")

    except PlannerOutputError as exc:
        print("\n===== PLANNER PARSE ERROR =====")
        print(exc)

        if recording_model.last_response is not None:
            print("\n===== RAW MODEL OUTPUT =====")
            print(recording_model.last_response.text)

        print("\nPLANNER E2E: FAILED")
        raise


if __name__ == "__main__":
    main()
