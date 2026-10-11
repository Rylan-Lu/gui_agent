"""Week 6 REAL desktop A/B trial runner (Windows only).

One trial per command. Run only on a controlled desktop after explicit --execute.
Baseline: grounding retry=1, Recovery/Replan disabled.
Enhanced: grounding retry=2, Recovery enabled; Replan is opt-in.
Faults here are *injected during real GUI execution*, NOT natural UI failures.
Do not log full OCR contents or save screenshots by default.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "robustness" / "week6"
TARGET_APP = ROOT / "scripts" / "e2e" / "e2e_target_app.py"
TASKS = ("browser_open", "browser_search", "text_entry")
FAULTS = ("none", "grounding_once", "grounding_exhausted", "grounding_replan_window", "post_action_once")
MODES = ("baseline", "enhanced")


@dataclass(frozen=True)
class TrialSpec:
    task: str
    mode: str
    trial: int
    fault: str = "none"
    replan_backend: str = "none"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError("unknown task")
        if self.mode not in MODES:
            raise ValueError("unknown mode")
        if type(self.trial) is not int or self.trial < 1:
            raise ValueError("trial must be >= 1")
        if self.fault not in FAULTS:
            raise ValueError("unknown fault")
        if self.fault in ("grounding_once", "grounding_exhausted", "grounding_replan_window") and self.task == "browser_open":
            raise ValueError("browser_open has no OCR-grounded action")
        if self.replan_backend not in ("none", "local-vlm"):
            raise ValueError("unknown replan backend")
        if self.replan_backend != "none" and self.mode != "enhanced":
            raise ValueError("Replan backend is only valid in enhanced mode")


@dataclass(frozen=True)
class TrialFixture:
    plan_data: dict[str, Any]
    marker: str
    receipt: Path | None = None
    html_file: Path | None = None
    query: str | None = None


def _plan_steps(actions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if not 1 <= len(actions) <= 8:
        raise ValueError("TaskPlan supports only 1 to 8 actions")
    return {
        "steps": [
            {"step_id": i, "description": description, "action": action}
            for i, (description, action) in enumerate(actions, start=1)
        ]
    }


def _step(kind: str, **kwargs: Any) -> dict[str, Any]:
    return {"action_type": kind, **kwargs}


def should_inject_grounding(fault: str, prior_injections: int) -> bool:
    """Keep permanent and initially-exhausted fault scenarios separate.

    grounding_replan_window injects exactly the first two misses needed
    to exhaust enhanced mode's configured Grounding retry budget.  It
    then allows newly planned Grounding actions to run normally.
    """
    if fault == "grounding_once":
        return prior_injections == 0
    if fault == "grounding_exhausted":
        return True
    if fault == "grounding_replan_window":
        return prior_injections < 2
    return False


def safe_failure_code(reason: str | None) -> str:
    """Summarize failures without logging model text, prompts or OCR."""
    if not reason:
        return "none"
    if "timeout exceeded" in reason:
        return "timeout"
    if "replan limit exceeded" in reason:
        return "replan_limit"
    if "no remaining step budget" in reason:
        return "step_budget"
    if "replan starts typing without restoring input focus" in reason:
        return "replan_missing_focus"
    if "new plan repeats the failed action" in reason:
        return "replan_repeats_failed_action"
    if "new plan repeats a completed action" in reason:
        return "replan_repeats_completed_action"
    if "new plan exceeds remaining step budget" in reason:
        return "replan_exceeds_budget"
    if "non-executable step" in reason:
        return "replan_missing_action"
    if "PlannerOutputError" in reason:
        return "planner_output_invalid"
    if "replanning failed:" in reason:
        tail = reason.rsplit("replanning failed:", 1)[-1]
        error_class = tail.split(":", 1)[0].strip()
        known_types = {
            "ReplanError", "RuntimeError", "ValueError", "TypeError",
            "OSError", "CUDAOutOfMemoryError", "OutOfMemoryError",
        }
        return (f"replan_{error_class}" if error_class in known_types
                else "replan_exception")
    if "injected pre-dispatch target miss" in reason:
        return "grounding_miss"
    return "other_failure"


def safe_judge_code(result: Any) -> str:
    """Report final Judge outcome without retaining OCR or task text."""
    if result is None:
        return "runner_exception"
    judge = getattr(result, "judge_result", None)
    if judge is None:
        phase = getattr(getattr(result, "failure_decision", None), "phase", None)
        if getattr(phase, "value", "") == "success_judge":
            return "judge_exception"
        return "not_run"
    if getattr(judge, "success", False):
        return "passed"
    reason = getattr(judge, "reason", "") or ""
    if "local receipt missing" in reason:
        return "receipt_missing"
    if "local receipt mismatch" in reason:
        return "receipt_mismatch"
    if "no OCR results available" in reason:
        return "ocr_unavailable"
    if "expected text not found" in reason:
        return "visual_marker_missing"
    return "judge_rejected_other"


def safe_action_types(plan: Any) -> list[str]:
    """Only return enum action names; never copy model-generated text."""
    steps = getattr(plan, "steps", ())
    result = []
    for step in steps:
        action = getattr(step, "action", None)
        action_type = getattr(getattr(action, "action_type", None), "value", None)
        result.append(action_type if action_type in {
            "click", "double_click", "type_text", "key_press",
            "hotkey", "scroll", "wait"
        } else "unknown")
    return result


def safe_step_outcomes(runtime: Any) -> list[str]:
    """Return success flags only, never StepRecord.description/error."""
    history = getattr(getattr(runtime, "state", None), "history", ())
    return ["pass" if step.success is True else
            "fail" if step.success is False else "unknown"
            for step in history]


def build_fixture(spec: TrialSpec, directory: Path) -> TrialFixture:
    """Produce isolated local files and a plan; does NOT touch the desktop."""
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10].upper()
    marker = f"W6_{spec.task.upper()}_{token}"

    if spec.task in ("browser_open", "browser_search"):
        page = directory / f"{spec.task}_{spec.mode}_{spec.trial}_{token}.html"
        if spec.task == "browser_open":
            document = (
                '<!doctype html><html><head><meta charset="utf-8">'
                '<title>GUI Agent Week 6 Browser Fixture</title>'
                '</head><body style="font:42px Segoe UI,sans-serif;padding:42px">'
                f'<h1>{html.escape(marker)}</h1></body></html>'
            )
        else:
            query = f"W6QUERY_{token}"
            # The completion marker is in JS source, never visible until a real submit.
            js_query = json.dumps(query)
            js_marker = json.dumps(marker)
            document = (
                '<!doctype html><html><head><meta charset="utf-8">'
                '<title>GUI Agent Week 6 Local Search</title>'
                '<style>body{font:38px Segoe UI,sans-serif;padding:44px}'
                'label,input,button{font-size:34px;display:block;margin:16px 0}'
                'input{width:750px;padding:12px}button{padding:12px}</style></head><body>'
                '<h1>LOCAL SEARCH TEST</h1><form id="form">'
                '<label for="q">SEARCH_QUERY</label>'
                '<input id="q" type="text" autocomplete="off">'
                '<button type="submit">SEARCH</button></form>'
                '<h2 id="result"></h2>'
                '<script>document.getElementById("form").addEventListener("submit",'
                'function(e){e.preventDefault();let typed=document.getElementById("q").value;'
                f'document.getElementById("result").textContent=(typed==={js_query})?'
                f'{js_marker}:"NO_RESULT";}});</script></body></html>'
            )
        page.write_text(document, encoding="utf-8")
        command = f'msedge.exe --new-window "{page.resolve().as_uri()}"'
        actions = [
            ("Open Windows Run dialog", _step("hotkey", keys=["winleft", "r"])),
            ("Wait for Run dialog", _step("wait", wait_seconds=0.7)),
            ("Enter local Edge fixture command", _step("type_text", text=command)),
            ("Launch Edge", _step("key_press", keys=["enter"])),
            ("Wait for local page", _step("wait", wait_seconds=2.0)),
        ]
        if spec.task == "browser_search":
            actions.extend([
                ("Focus local search input through its visible label", _step("click", element="SEARCH_QUERY")),
                ("Type unique search query", _step("type_text", text=query)),
                ("Submit local search", _step("key_press", keys=["enter"])),
            ])
        return TrialFixture(
            plan_data=_plan_steps(actions), marker=marker, html_file=page,
            query=query if spec.task == "browser_search" else None,
        )

    receipt = directory / f"receipt_{spec.mode}_{spec.trial}_{token}.txt"
    return TrialFixture(
        plan_data=_plan_steps([
            ("Focus local message field", _step("click", element="E2E_TARGET")),
            ("Enter unique test marker", _step("type_text", text=marker)),
            ("Click local SEND button", _step("click", element="SEND")),
            ("Wait for receipt", _step("wait", wait_seconds=0.8)),
        ]),
        marker=marker,
        receipt=receipt,
    )


def build_runtime_task(spec: TrialSpec, fixture: TrialFixture) -> str:
    """Give the Replanner an exact non-sensitive task objective."""
    if spec.task == "text_entry":
        return (
            "In the GUI Agent Message Target window, focus E2E_TARGET, "
            f"type exactly {fixture.marker} into the message field, "
            "activate SEND, and confirm MESSAGE SENT with the exact "
            "matching local receipt."
        )
    if spec.task == "browser_search":
        return (
            "On the local browser test page, focus SEARCH_QUERY, "
            f"enter exactly {fixture.query}, submit the local search, "
            f"and confirm the result marker {fixture.marker} is visible."
        )
    return (
        "Open the local test page in Edge and confirm the visible "
        f"marker {fixture.marker}."
    )


def make_record(
    spec: TrialSpec, *, success: bool, human_success: bool | None,
    duration_s: float, setup_duration_s: float, steps: int,
    failure_category: str, recovery_status: str, replan_calls: int,
    injected_count: int, exception_type: str | None = None,
) -> dict[str, Any]:
    """Compatible with week6_real_record.py. No OCR text or screenshot paths."""
    if duration_s < 0 or setup_duration_s < 0 or steps < 0:
        raise ValueError("negative metric")
    return {
        "source": "real_desktop_controlled" if spec.fault != "none" else "real_desktop_natural",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": f"{spec.task}/{spec.fault}",  # avoids mixing injected and natural samples
        "task_base": spec.task,
        "trial": spec.trial,
        "mode": spec.mode,
        "fault": spec.fault,
        "replan_backend": spec.replan_backend,
        "auto_success": bool(success),
        "human_success": human_success,
        "duration_s": round(duration_s, 4),
        "setup_duration_s": round(setup_duration_s, 4),
        "steps": steps,
        "failure_category": failure_category,
        "recovery_status": recovery_status,
        "replan_calls": replan_calls,
        "injected_count": injected_count,
        "exception_type": exception_type,
    }


def append_record(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def ask_human_review(choice: str) -> bool | None:
    if choice == "yes":
        return True
    if choice == "no":
        return False
    if choice == "unknown":
        return None
    while True:
        raw = input("人工查看当前真实桌面：任务是否完成？ [y/n/u]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        if raw in ("u", "unknown", ""):
            return None
        print("请输入 y、n 或 u。")


def run_trial(
    spec: TrialSpec, *, directory: Path, countdown: float,
    review: str, ocr_device: str = "gpu:0", timeout_s: float = 90.0,
) -> dict[str, Any]:
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300):
        raise ValueError("timeout_s must be finite and within (0, 300]")
    if ocr_device not in ("cpu", "gpu:0"):
        raise ValueError("ocr_device must be 'cpu' or 'gpu:0'")
    if os.name != "nt":
        raise RuntimeError("Real desktop execution requires Windows")
    if not TARGET_APP.is_file() and spec.task == "text_entry":
        raise FileNotFoundError(TARGET_APP)

    # Avoid importing Paddle before establishing the known Torch-first runtime.
    from gui_agent.runtime.gpu_runtime import initialize_torch_gpu_runtime
    initialize_torch_gpu_runtime()

    from gui_agent.agent.action_runner import ActionRunner
    from gui_agent.agent.action_transaction import ActionTransaction
    from gui_agent.agent.environment import DesktopEnvironment
    from gui_agent.agent.executor import ActionExecutor
    from gui_agent.agent.feedback import ActionFeedback
    from gui_agent.agent.grounding import GroundingBridge, GroundingTargetNotFoundError
    from gui_agent.agent.loop import AgentLoop
    from gui_agent.agent.plan_schema import TaskPlan
    from gui_agent.agent.recovery import RecoveryEngine
    from gui_agent.agent.replan import Replanner
    from gui_agent.agent.retry import RetryPolicy
    from gui_agent.agent.runtime import AgentRuntime
    from gui_agent.agent.success_judge import SuccessJudgeResult, TextSuccessJudge
    from gui_agent.control.controller import Controller
    from gui_agent.ocr.paddleocr_engine import PaddleOCREngine

    class OnceGroundingFault(GroundingBridge):
        count = 0
        def ground(self, action, **kwargs):
            if should_inject_grounding(spec.fault, self.count):
                self.count += 1
                raise GroundingTargetNotFoundError("injected pre-dispatch target miss")
            return super().ground(action, **kwargs)

    class OnceFeedbackFault(ActionFeedback):
        count = 0
        def compare(self, before, after):
            self.count += 1
            if spec.fault == "post_action_once" and self.count == len(fixture.plan_data["steps"]):
                raise RuntimeError("injected post-action feedback error")
            return super().compare(before, after)

    class CountedReplanner(Replanner):
        calls = 0
        validated = False
        action_types: list[str] = []
        error_type: str | None = None

        def replan(self, *args, **kwargs):
            self.calls += 1
            try:
                plan = super().replan(*args, **kwargs)
            except Exception as exc:
                self.error_type = type(exc).__name__
                raise
            self.validated = True
            self.action_types = safe_action_types(plan)
            return plan

    fixture = build_fixture(spec, directory / "fixtures")
    target_process = None
    environment = None
    result = None
    error_type = None
    duration_s = 0.0
    setup_start = time.perf_counter()
    grounding = OnceGroundingFault()
    feedback = OnceFeedbackFault()
    counted_replanner = None
    model = None
    human_success: bool | None = None
    try:
        if spec.task == "text_entry":
            # Re-use the existing owned local target with a receipt file.
            if fixture.receipt.is_file():
                fixture.receipt.unlink()
            target_process = subprocess.Popen([
                sys.executable, str(TARGET_APP), "--receipt", str(fixture.receipt.resolve())
            ])
            time.sleep(1.5)

        if spec.replan_backend == "local-vlm":
            from gui_agent.models.local_vlm import LocalVLMClient
            from gui_agent.agent.planner import Planner
            # Limit generated tokens to reduce KV-cache memory during VLM Replan.
            model = LocalVLMClient(max_new_tokens=256)
            counted_replanner = CountedReplanner(
                Planner(model), max_replans=1,
                require_focus_after_failed_click=(spec.task == "text_entry"),
            )

        environment = DesktopEnvironment(
            ocr_engine=PaddleOCREngine(lang="ch", device=ocr_device, min_confidence=0.5)
        )
        transaction = ActionTransaction(
            environment, ActionRunner(ActionExecutor(Controller()), grounding=grounding),
            feedback=feedback, settle_seconds=0.5,
            retry_policy=RetryPolicy(
                max_grounding_attempts=2 if spec.mode == "enhanced" else 1,
                grounding_retry_delay_s=0.35,
            ),
        )
        runtime = AgentRuntime.create(
            task=build_runtime_task(spec, fixture),
            max_steps=10, timeout_s=float(timeout_s),
        )
        loop = AgentLoop(
            runtime, transaction,
            recovery_engine=RecoveryEngine(environment) if spec.mode == "enhanced" else None,
            replanner=counted_replanner,
        )
        judge = TextSuccessJudge()

        def review_observation(observation) -> SuccessJudgeResult:
            visual = judge.judge(observation, fixture.marker if spec.task != "text_entry" else "MESSAGE SENT")
            if not visual.success or spec.task != "text_entry":
                return visual
            if fixture.receipt is None or not fixture.receipt.is_file():
                return SuccessJudgeResult(False, "local receipt missing")
            if fixture.receipt.read_text(encoding="utf-8").strip() != fixture.marker:
                return SuccessJudgeResult(False, "local receipt mismatch")
            return SuccessJudgeResult(True, "visual sent indicator and local receipt agree")

        def final_check() -> SuccessJudgeResult:
            return review_observation(environment.observe(use_ocr=True))

        setup_duration_s = time.perf_counter() - setup_start
        print(f"REAL DESKTOP | {spec.mode} | {spec.task} | fault={spec.fault} | trial={spec.trial}")
        print("请切换到可操作的 Windows 桌面；本次会主动发送鼠标和键盘事件。")
        if spec.task != "text_entry":
            print("浏览器窗口不会由脚本强制关闭；请在试次结束后手动关闭本次测试窗口。")
        if spec.replan_backend == "none":
            print("本次未启用模型 Replan（仅对比 Retry/Recovery）。")
        for second in range(int(countdown), 0, -1):
            print(f"Starting in {second}...", flush=True)
            time.sleep(1)
        if countdown > int(countdown):
            time.sleep(countdown - int(countdown))

        start = time.perf_counter()
        try:
            result = loop.run(
                TaskPlan.from_dict(fixture.plan_data),
                final_check=final_check,
                recovery_check=review_observation if spec.mode == "enhanced" else None,
            )
        except Exception as exc:
            error_type = type(exc).__name__
            print(f"Runner exception type: {error_type}")
        duration_s = time.perf_counter() - start
        if result is not None:
            print(f"Auto judge: {'PASS' if result.success else 'FAIL'}")
            print(f"Steps attempted: {runtime.state.step_count} | duration={duration_s:.2f}s")
            print("Recovery:", getattr(getattr(result, "recovery_result", None), "status", "none"))
            print("Replan calls:", counted_replanner.calls if counted_replanner is not None else 0)
            print("Replan validated:", bool(counted_replanner and counted_replanner.validated))
            print("Replan action types:", (counted_replanner.action_types
                                            if counted_replanner is not None else []))
            print("Step outcomes:", safe_step_outcomes(runtime))
            print("Final judge code:", safe_judge_code(result))
            if not result.success:
                print("Safe failure code:", safe_failure_code(result.failure_reason))
        print("请现在检查桌面实际状态。不会自动把未人工复核的任务标成失败。")
        human_success = ask_human_review(review)
        failure_category = (
            getattr(getattr(getattr(result, "failure_decision", None), "phase", None), "value", "none")
            if result is not None else "runner_exception"
        )
        recovery_status = (
            getattr(getattr(getattr(result, "recovery_result", None), "status", None), "value", "none")
            if result is not None else "none"
        )
        row = make_record(
            spec, success=bool(result is not None and result.success),
            human_success=human_success, duration_s=duration_s,
            setup_duration_s=setup_duration_s, steps=runtime.state.step_count,
            failure_category=failure_category, recovery_status=recovery_status,
            replan_calls=counted_replanner.calls if counted_replanner is not None else 0,
            injected_count=grounding.count if spec.fault in ("grounding_once", "grounding_exhausted", "grounding_replan_window") else (
                1 if spec.fault == "post_action_once" and feedback.count >= len(fixture.plan_data["steps"])
                else 0
            ),
            exception_type=error_type,
        )
        row["failure_code"] = safe_failure_code(
            result.failure_reason if result is not None else None
        ) if result is not None else "runner_exception"
        row["final_judge_code"] = safe_judge_code(result)
        row["step_outcomes"] = safe_step_outcomes(runtime)
        row["replan_validated"] = bool(counted_replanner and counted_replanner.validated)
        row["replan_action_types"] = (
            counted_replanner.action_types if counted_replanner is not None else []
        )
        row["replan_error_type"] = (
            counted_replanner.error_type if counted_replanner is not None else None
        )
        row["ocr_device"] = ocr_device
        row["timeout_s"] = float(timeout_s)
        append_record(directory / "real_runs.jsonl", row)
        print("Recorded:", (directory / "real_runs.jsonl").resolve())
        return row
    finally:
        # Defensive fallback if one-shot planning did not reach Replanner's
        # own cleanup (e.g. a runner-level exception). This is safe for an
        # uninitialized LocalVLMClient as well.
        if model is not None:
            model.unload()
        if environment is not None:
            environment.close()
        if target_process is not None and target_process.poll() is None:
            target_process.terminate()
            try:
                target_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                target_process.kill()
                target_process.wait(timeout=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--fault", choices=FAULTS, default="none")
    parser.add_argument("--replan-backend", choices=("none", "local-vlm"), default="none")
    parser.add_argument("--ocr-device", choices=("gpu:0", "cpu"), default="gpu:0",
                        help="OCR device. Use cpu for memory-isolated local VLM experiments.")
    parser.add_argument("--timeout-s", type=float, default=90.0,
                        help="Task budget in seconds; 90 by default, up to 300 for diagnosis.")
    parser.add_argument("--review", choices=("prompt", "yes", "no", "unknown"), default="prompt")
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--execute", action="store_true", help="Explicit consent to operate the real Windows desktop")
    args = parser.parse_args(argv)
    try:
        spec = TrialSpec(args.task, args.mode, args.trial, args.fault, args.replan_backend)
    except ValueError as exc:
        parser.error(str(exc))
    if args.countdown < 0 or not math.isfinite(float(args.countdown)):
        parser.error("countdown must be finite and nonnegative")
    if not math.isfinite(args.timeout_s) or not 0 < args.timeout_s <= 300:
        parser.error("--timeout-s must be finite and within (0, 300]")
    if not args.execute:
        print("DRY RUN: no desktop actions. Add --execute to operate Windows GUI.")
        print("Trial:", spec)
        return 0
    run_trial(spec, directory=args.output, countdown=args.countdown,
              review=args.review, ocr_device=args.ocr_device,
              timeout_s=args.timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
