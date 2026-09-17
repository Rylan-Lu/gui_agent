from __future__ import annotations

from pathlib import Path
from time import perf_counter

from gui_agent.agent.chain import build_planning_chain
from gui_agent.agent.planner import Planner
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.models.local_vlm import LocalVLMClient


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


def main():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    instruction = (
        "In the currently open Firefox browser, "
        "search for von Neumann and review "
        "the search results."
    )

    print("===== AGENT CHAIN E2E =====")

    # 1. Initialize local VLM.
    model = LocalVLMClient(
        max_new_tokens=384
    )

    # 2. Create Planner.
    planner = Planner(model=model)

    # 3. Build LangChain pipeline.
    chain = build_planning_chain(planner)

    # 4. Invoke the complete planning chain.
    start = perf_counter()

    result = chain.invoke({
        "instruction": instruction,
        "screenshot_path": IMAGE_PATH,
    })

    elapsed = perf_counter() - start

    # 5. Validate output.
    assert isinstance(result, TaskPlan)
    assert len(result.steps) > 0

    print("\n===== TASK PLAN =====")

    for step in result.steps:
        print(
            f"{step.step_id}. {step.description}"
        )

    print("\n===== RESULT =====")
    print("Steps:", len(result.steps))
    print(f"Total time: {elapsed:.2f}s")

    print("\nAGENT CHAIN E2E: PASS")


if __name__ == "__main__":
    main()