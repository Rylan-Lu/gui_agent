from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from gui_agent.agent.planner import (
    Planner,
    PlannerOutputError,
)
from gui_agent.models.local_vlm import LocalVLMClient
from gui_agent.models.base import (
    ModelRequest,
    ModelResponse,
)


ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT.parent
    / "ScreenAgent_reference"
    / "data"
    / "ScreenAgent"
    / "train"
    / "02ea503d419c440cbda9e42263706d6a"
    / "images"
    / "2023-12-20_19-31-31-255913.jpg"
)


class RecordingModel:
    """Record the raw model response for debugging."""

    def __init__(self, model):
        self.model = model
        self.last_response = None

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:

        response = self.model.generate(request)
        self.last_response = response

        return response


def main():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    print("===== REAL PLANNER TEST =====")
    print("Screenshot:", IMAGE_PATH)

    instruction = (
        "In the currently open Firefox browser, "
        "search for von Neumann and review "
        "the search results."
    )

    print("\nTask:", instruction)

    # 1. Create the real local VLM.
    local_model = LocalVLMClient(
        max_new_tokens=384,
    )

    # 2. Record the raw output for debugging.
    recording_model = RecordingModel(local_model)

    # 3. Inject the model into Planner.
    planner = Planner(
        model=recording_model,
    )

    # 4. Generate and validate the plan.
    start = perf_counter()

    try:

        plan = planner.plan(
            instruction=instruction,
            screenshot_path=IMAGE_PATH,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = perf_counter() - start

        print("\n===== RAW MODEL OUTPUT =====")

        print(recording_model.last_response.text)

        print("\n===== VALIDATED TASK PLAN =====")

        for step in plan.steps:
            print(
                f"{step.step_id}. "
                f"{step.description}"
            )

        print("\nStep count:", len(plan.steps))
        print(f"Total time: {elapsed:.2f}s")

        print("\nPLANNER E2E: PASS")

    except PlannerOutputError as exc:

        print("\n===== PLANNER PARSE ERROR =====")
        print(exc)

        if recording_model.last_response is not None:

            print("\n===== RAW MODEL OUTPUT =====")

            print(
                recording_model.last_response.text
            )

        print("\nPLANNER E2E: FAILED")

        raise


if __name__ == "__main__":
    main()