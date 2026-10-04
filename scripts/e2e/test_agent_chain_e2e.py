from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from gui_agent.agent.chain import build_planning_chain
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.planner import Planner
from gui_agent.models.local_vlm import LocalVLMClient


DEFAULT_INSTRUCTION = (
    "In the currently open Firefox browser, "
    "search for von Neumann and review the search results."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real screenshot -> local VLM -> Planner -> "
            "LangChain planning path."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Screenshot used by the planning chain.",
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

    print("===== AGENT CHAIN E2E =====")
    print("Screenshot:", image_path)
    print("Task:", args.instruction)

    model = LocalVLMClient(
        max_new_tokens=args.max_new_tokens,
    )
    planner = Planner(model=model)
    chain = build_planning_chain(planner)

    start = perf_counter()

    result = chain.invoke({
        "instruction": args.instruction,
        "screenshot_path": image_path,
    })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = perf_counter() - start

    if not isinstance(result, TaskPlan):
        raise RuntimeError(
            f"Planning chain returned {type(result).__name__}, "
            "expected TaskPlan"
        )

    if not result.steps:
        raise RuntimeError("Planning chain returned an empty TaskPlan")

    print("\n===== TASK PLAN =====")
    for step in result.steps:
        print(f"{step.step_id}. {step.description}")

    print("\n===== RESULT =====")
    print("Steps:", len(result.steps))
    print(f"Total time: {elapsed:.2f}s")
    print("\nAGENT CHAIN E2E: PASS")


if __name__ == "__main__":
    main()
