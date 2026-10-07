from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.desktop_agent import DesktopAgent
from gui_agent.agent.environment import DesktopEnvironment
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.planner import Planner
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
    TextSuccessJudge,
)
from gui_agent.control.controller import Controller
from gui_agent.models.local_vlm import LocalVLMClient
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

ROOT = Path(__file__).resolve().parents[2]
TARGET_APP = ROOT / "scripts" / "e2e" / "e2e_target_app.py"
ARTIFACT_DIR = ROOT / "artifacts" / "e2e" / "week4"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    marker = f"W4UNIFIED_{int(time.time())}"
    receipt = ARTIFACT_DIR / "unified_agent_receipt.txt"

    if receipt.exists():
        receipt.unlink()

    target_process: subprocess.Popen[bytes] | None = None
    environment: DesktopEnvironment | None = None

    try:
        target_process = subprocess.Popen(
            [
                sys.executable,
                str(TARGET_APP),
                "--receipt",
                str(receipt.resolve()),
            ],
        )
        time.sleep(1.5)

        model = LocalVLMClient(
            max_new_tokens=384,
        )
        planner = Planner(model=model)

        ocr = PaddleOCREngine(
            lang="ch",
            device="gpu:0",
            min_confidence=0.5,
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
            planner=planner,
            environment=environment,
            transaction=transaction,
            max_steps=8,
            timeout_s=45.0,
        )

        instruction = (
            "In the visible GUI Agent Message Target window, "
            "click E2E_TARGET to focus the message input, "
            f"type exactly {marker}, click SEND, and wait until "
            "MESSAGE SENT is visible."
        )

        text_judge = TextSuccessJudge()

        def final_check() -> SuccessJudgeResult:
            observation = environment.observe(use_ocr=True)
            visual = text_judge.judge(
                observation,
                "MESSAGE SENT",
            )

            if not visual.success:
                return visual

            if not receipt.is_file():
                return SuccessJudgeResult(
                    success=False,
                    reason="message receipt was not created",
                )

            received = receipt.read_text(
                encoding="utf-8"
            ).strip()

            if received != marker:
                return SuccessJudgeResult(
                    success=False,
                    reason=(
                        "message receipt mismatch: "
                        f"expected {marker!r}, got {received!r}"
                    ),
                )

            return SuccessJudgeResult(
                success=True,
                reason=(
                    "MESSAGE SENT was visible and the exact receipt matched"
                ),
                matched_text=visual.matched_text,
            )

        print("===== UNIFIED DESKTOP AGENT E2E =====")
        print("Task:", instruction)

        result = agent.run(
            instruction,
            final_check=final_check,
        )

        print("\n===== TASK PLAN =====")
        for step in result.plan.steps:
            print(f"{step.step_id}. {step.description}")

        print("\n===== RESULT =====")
        print("success:", result.success)
        print("steps:", len(result.execution.transactions))
        print(f"capture_time: {result.capture_seconds:.2f}s")
        print(f"planning_time: {result.planning_seconds:.2f}s")
        print(f"execution_time: {result.execution_seconds:.2f}s")
        print(f"total_time: {result.total_seconds:.2f}s")

        if not result.success:
            raise RuntimeError(
                result.execution.failure_reason
                or "unified desktop agent failed"
            )

        print("\nUNIFIED DESKTOP AGENT E2E: PASS")

    finally:
        if environment is not None:
            environment.close()

        if (
            target_process is not None
            and target_process.poll() is None
        ):
            target_process.terminate()
            try:
                target_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                target_process.kill()
                target_process.wait(timeout=3.0)


if __name__ == "__main__":
    main()
