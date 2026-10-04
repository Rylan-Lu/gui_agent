from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.environment import DesktopEnvironment
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import (
    SuccessJudgeResult,
    TextAbsentJudge,
    TextSuccessJudge,
)
from gui_agent.control.controller import Controller
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

ROOT = Path(__file__).resolve().parents[2]
E2E_ARTIFACT_DIR = ROOT / "artifacts" / "e2e" / "week4"

BROWSER_MARKER = "WEEK4_BROWSER_OPEN_PASS"
FILE_MARKER = "WEEK4_FILE_OPEN_PASS"

BROWSER_FILE = E2E_ARTIFACT_DIR / "week4_browser_test.html"
TEST_FILE = E2E_ARTIFACT_DIR / "week4_test_file.txt"
TARGET_APP = ROOT / "scripts" / "e2e" / "e2e_target_app.py"


def prepare_test_artifacts() -> None:
    """Create temporary files used by the real desktop task suite."""

    E2E_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    BROWSER_FILE.write_text(
        f"""
        <html>
        <body>
            <h1>{BROWSER_MARKER}</h1>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    TEST_FILE.write_text(
        FILE_MARKER,
        encoding="utf-8",
    )


def build_plan(steps):
    return TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": index,
                    **step,
                }
                for index, step in enumerate(
                    steps,
                    start=1,
                )
            ]
        }
    )


def main():
    prepare_test_artifacts()
    message_marker = f"W4MSG_{int(time.time())}"
    message_receipt = E2E_ARTIFACT_DIR / "message_receipt.txt"

    if message_receipt.exists():
        message_receipt.unlink()

    ocr = PaddleOCREngine(
        lang="ch",
        device="gpu:0",
        min_confidence=0.5,
    )

    environment = DesktopEnvironment(
        ocr_engine=ocr,
    )

    controller = Controller()

    transaction = ActionTransaction(
        environment,
        ActionRunner(
            ActionExecutor(controller)
        ),
        settle_seconds=0.5,
    )

    text_judge = TextSuccessJudge()
    absent_judge = TextAbsentJudge()

    results = []
    message_process: subprocess.Popen[bytes] | None = None

    def run_task(
            name,
            plan,
            final_check,
            max_steps=12,
    ):
        runtime = AgentRuntime.create(
            task=name,
            max_steps=max_steps,
            timeout_s=45,
        )

        loop = AgentLoop(
            runtime,
            transaction,
        )

        start_time = time.perf_counter()

        result = loop.run(
            plan,
            final_check=final_check,
        )

        elapsed = time.perf_counter() - start_time

        changed_steps = sum(
            1
            for tx in result.transactions
            if tx.feedback.changed
        )

        results.append(
            {
                "name": name,
                "success": result.success,
                "steps": runtime.state.step_count,
                "duration_s": elapsed,
                "changed_steps": changed_steps,
                "failure_reason": result.failure_reason,
            }
        )

        print(
            f"{name}: "
            f"{'PASS' if result.success else 'FAIL'} "
            f"| steps={runtime.state.step_count} "
            f"| time={elapsed:.2f}s "
            f"| changed={changed_steps}"
        )

    try:
        # 1. 浏览器
        browser_url = BROWSER_FILE.resolve().as_uri()

        run_task(
            "01 Open Browser",
            build_plan(
                [
                    {
                        "description": "Open Run dialog",
                        "action": {
                            "action_type": "hotkey",
                            "keys": ["winleft", "r"],
                        },
                    },
                    {
                        "description": "Wait",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 0.8,
                        },
                    },
                    {
                        "description": "Open browser test page",
                        "action": {
                            "action_type": "type_text",
                            "text": f'msedge.exe "{browser_url}"',
                        },
                    },
                    {
                        "description": "Launch Edge",
                        "action": {
                            "action_type": "key_press",
                            "keys": ["enter"],
                        },
                    },
                    {
                        "description": "Wait for Edge",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 3.0,
                        },
                    },
                ]
            ),
            lambda: text_judge.judge(
                environment.observe(use_ocr=True),
                BROWSER_MARKER,
            ),
        )

        # 2. 浏览器搜索
        query = f"GUI_AGENT_WEEK4_{int(time.time())}"

        run_task(
            "02 Browser Search",
            build_plan(
                [
                    {
                        "description": "Focus address bar",
                        "action": {
                            "action_type": "hotkey",
                            "keys": ["ctrl", "l"],
                        },
                    },
                    {
                        "description": "Type search query",
                        "action": {
                            "action_type": "type_text",
                            "text": query,
                        },
                    },
                    {
                        "description": "Search",
                        "action": {
                            "action_type": "key_press",
                            "keys": ["enter"],
                        },
                    },
                    {
                        "description": "Wait for results",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 3.0,
                        },
                    },
                ]
            ),
            lambda: text_judge.judge(
                environment.observe(use_ocr=True),
                query,
            ),
        )

        # 3. 打开文件
        command = f'notepad.exe "{TEST_FILE}"'

        run_task(
            "03 Open File",
            build_plan(
                [
                    {
                        "description": "Open Run dialog",
                        "action": {
                            "action_type": "hotkey",
                            "keys": ["winleft", "r"],
                        },
                    },
                    {
                        "description": "Wait",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 0.8,
                        },
                    },
                    {
                        "description": "Type file command",
                        "action": {
                            "action_type": "type_text",
                            "text": command,
                        },
                    },
                    {
                        "description": "Open file",
                        "action": {
                            "action_type": "key_press",
                            "keys": ["enter"],
                        },
                    },
                    {
                        "description": "Wait for Notepad",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 2.0,
                        },
                    },
                ]
            ),
            lambda: text_judge.judge(
                environment.observe(use_ocr=True),
                FILE_MARKER,
            ),
        )

        # 4. 发送消息
        # Launch a deterministic local target instead of depending on a
        # third-party messaging client, account, network, or changing UI.
        message_process = subprocess.Popen(
            [
                sys.executable,
                str(TARGET_APP),
                "--receipt",
                str(message_receipt.resolve()),
            ],
        )
        time.sleep(1.5)

        def sent_check() -> SuccessJudgeResult:
            observation = environment.observe(use_ocr=True)
            visual_result = text_judge.judge(
                observation,
                "MESSAGE SENT",
            )

            if not visual_result.success:
                return visual_result

            if not message_receipt.exists():
                return SuccessJudgeResult(
                    success=False,
                    reason="message receipt was not created",
                )

            received = message_receipt.read_text(
                encoding="utf-8"
            ).strip()

            if received != message_marker:
                return SuccessJudgeResult(
                    success=False,
                    reason=(
                        "message receipt mismatch: "
                        f"expected {message_marker!r}, got {received!r}"
                    ),
                )

            return SuccessJudgeResult(
                success=True,
                reason=(
                    "message target displayed sent state and "
                    "receipt matched"
                ),
                matched_text=visual_result.matched_text,
            )

        run_task(
            "04 Send Message",
            build_plan(
                [
                    {
                        "description": "Focus message input",
                        "action": {
                            "action_type": "click",
                            "element": "E2E_TARGET",
                        },
                    },
                    {
                        "description": "Type message",
                        "action": {
                            "action_type": "type_text",
                            "text": message_marker,
                        },
                    },
                    {
                        "description": "Send message",
                        "action": {
                            "action_type": "click",
                            "element": "SEND",
                        },
                    },
                    {
                        "description": "Wait for sent status",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 0.8,
                        },
                    },
                ]
            ),
            sent_check,
            max_steps=8,
        )

        # 5. 关闭应用
        def closed_check() -> SuccessJudgeResult:
            observation = environment.observe(use_ocr=True)
            absent_result = absent_judge.judge(
                observation,
                "MESSAGE SENT",
            )

            if message_process is not None and message_process.poll() is None:
                return SuccessJudgeResult(
                    success=False,
                    reason="message target process is still running",
                )

            if not absent_result.success:
                return absent_result

            return SuccessJudgeResult(
                success=True,
                reason="message target closed and sent marker is absent",
            )

        run_task(
            "05 Close Application",
            build_plan(
                [
                    {
                        "description": "Close message target",
                        "action": {
                            "action_type": "hotkey",
                            "keys": ["alt", "f4"],
                        },
                    },
                    {
                        "description": "Wait for close",
                        "action": {
                            "action_type": "wait",
                            "wait_seconds": 1.5,
                        },
                    },
                ]
            ),
            closed_check,
        )

    finally:
        if message_process is not None and message_process.poll() is None:
            message_process.terminate()

            try:
                message_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                message_process.kill()
                message_process.wait(timeout=3.0)

        environment.close()

    print("\n=== Week 4 Task Suite ===")

    passed = 0

    print("\n=== Week 4 Baseline ===")

    passed = sum(
        1
        for item in results
        if item["success"]
    )

    total = len(results)

    total_time = sum(
        item["duration_s"]
        for item in results
    )

    total_steps = sum(
        item["steps"]
        for item in results
    )

    for item in results:
        status = (
            "PASS"
            if item["success"]
            else "FAIL"
        )

        print(
            f"{status} | "
            f"{item['name']} | "
            f"steps={item['steps']} | "
            f"time={item['duration_s']:.2f}s | "
            f"changed={item['changed_steps']}"
        )

        if item["failure_reason"]:
            print(
                f"       reason="
                f"{item['failure_reason']}"
            )

    success_rate = (
        passed / total
        if total
        else 0.0
    )

    print("\n=== Summary ===")
    print(f"tasks: {total}")
    print(f"passed: {passed}")
    print(f"failed: {total - passed}")
    print(f"success_rate: {success_rate:.1%}")
    print(f"total_steps: {total_steps}")
    print(f"total_time: {total_time:.2f}s")

    if total:
        print(
            f"avg_steps: "
            f"{total_steps / total:.2f}"
        )

        print(
            f"avg_time: "
            f"{total_time / total:.2f}s"
        )


if __name__ == "__main__":
    main()