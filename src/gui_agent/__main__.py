from __future__ import annotations

import argparse
import sys
from typing import Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gui_agent",
        description=(
            "Plan and execute one natural-language task on the current "
            "Windows desktop."
        ),
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Natural-language desktop task.",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "api"),
        default="local",
        help="Planning model backend (default: local).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model ID. Local default: Qwen/Qwen3-VL-2B-Instruct. "
            "Required for --backend api."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "HTTPS Chat Completions endpoint. Required for --backend api."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="Maximum planner generation length for the local VLM.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of desktop execution steps.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Desktop execution timeout in seconds.",
    )
    parser.add_argument(
        "--ocr-device",
        default="gpu:0",
        help="PaddleOCR device (default: gpu:0).",
    )
    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=0.5,
        help="Minimum OCR confidence used for grounding.",
    )
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--expect-text",
        default=None,
        help="Require this text to be visible after execution.",
    )
    verification.add_argument(
        "--expect-absent",
        default=None,
        help="Require this text to be absent after execution.",
    )
    return parser


def _build_model(args: argparse.Namespace):
    if args.backend == "local":
        from gui_agent.models.local_vlm import LocalVLMClient

        model_id = args.model or "Qwen/Qwen3-VL-2B-Instruct"
        return LocalVLMClient(
            model_id=model_id,
            max_new_tokens=args.max_new_tokens,
        )

    if not args.endpoint:
        raise ValueError("--endpoint is required for --backend api")
    if not args.model:
        raise ValueError("--model is required for --backend api")

    from gui_agent.models.api_model import APIModelClient

    return APIModelClient(
        endpoint=args.endpoint,
        model_id=args.model,
        timeout=args.timeout,
    )


def _build_final_check(
    args: argparse.Namespace,
    environment,
) -> Callable | None:
    if args.expect_text is not None:
        from gui_agent.agent.success_judge import TextSuccessJudge

        judge = TextSuccessJudge()
        expected = args.expect_text
        return lambda: judge.judge(
            environment.observe(use_ocr=True),
            expected,
        )

    if args.expect_absent is not None:
        from gui_agent.agent.success_judge import TextAbsentJudge

        judge = TextAbsentJudge()
        forbidden = args.expect_absent
        return lambda: judge.judge(
            environment.observe(use_ocr=True),
            forbidden,
        )

    return None


def _print_result(result) -> None:
    print("\n=== TASK PLAN ===")
    for step in result.plan.steps:
        print(f"{step.step_id}. {step.description}")
        print(f"   action={step.action!r}")

    print("\n=== RESULT ===")
    print("execution_success:", result.success)
    print("steps:", len(result.execution.transactions))
    print(f"capture_time: {result.capture_seconds:.2f}s")
    print(f"planning_time: {result.planning_seconds:.2f}s")
    print(f"execution_time: {result.execution_seconds:.2f}s")
    print(f"total_time: {result.total_seconds:.2f}s")

    if result.execution.judge_result is None:
        print("verification: not requested")
    else:
        status = (
            "PASS"
            if result.execution.judge_result.success
            else "FAIL"
        )
        print("verification:", status)
        print(
            "verification_reason:",
            result.execution.judge_result.reason,
        )

    if result.execution.failure_reason:
        print(
            "failure_reason:",
            result.execution.failure_reason,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    environment = None

    try:
        model = _build_model(args)

        from gui_agent.agent.action_runner import ActionRunner
        from gui_agent.agent.action_transaction import ActionTransaction
        from gui_agent.agent.desktop_agent import DesktopAgent
        from gui_agent.agent.environment import DesktopEnvironment
        from gui_agent.agent.executor import ActionExecutor
        from gui_agent.agent.planner import Planner
        from gui_agent.control.controller import Controller
        from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

        ocr = PaddleOCREngine(
            lang="ch",
            device=args.ocr_device,
            min_confidence=args.ocr_confidence,
        )
        environment = DesktopEnvironment(
            ocr_engine=ocr,
        )
        transaction = ActionTransaction(
            environment,
            ActionRunner(
                ActionExecutor(
                    Controller()
                )
            ),
            settle_seconds=0.5,
        )
        agent = DesktopAgent(
            planner=Planner(model=model),
            environment=environment,
            transaction=transaction,
            max_steps=args.max_steps,
            timeout_s=args.timeout,
        )
        final_check = _build_final_check(
            args,
            environment,
        )

        print("=== GUI AGENT ===")
        print("backend:", args.backend)
        print("task:", args.task)
        print("Planning from the current desktop screenshot...")

        result = agent.run(
            args.task,
            final_check=final_check,
        )
        _print_result(result)

        return 0 if result.success else 1

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(
            f"GUI Agent failed: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
