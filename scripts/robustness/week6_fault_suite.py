"""Week 6 deterministic, MOCK-ONLY fault-injection comparison.

Imports the real AgentLoop / ActionTransaction / RecoveryEngine / Replanner.
No Controller, OCR model, GPU, network calls or desktop writes are performed.
Results MUST NOT be reported as real-desktop success rates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

# Direct `python scripts/robustness/week6_fault_suite.py` works without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gui_agent.agent.action_runner import ActionRunRequirements
from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.failure_policy import FailurePhase, decide_recovery
from gui_agent.agent.grounding import GroundingTargetNotFoundError
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.recovery import RecoveryEngine
from gui_agent.agent.replan import Replanner
from gui_agent.agent.retry import RetryPolicy
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.success_judge import SuccessJudgeResult
from gui_agent.datasets.schema import ActionType, GUIAction

SCENARIOS = (
    "delayed_grounding",
    "replan_after_miss",
    "dispatch_committed_then_error",
    "dispatch_unresolved",
    "post_action_error",
    "invalid_replan",
    "step_budget_exhausted",
    "replan_limit_exhausted",
    "planner_timeout",
)
MODES = ("baseline", "enhanced")
EXPECTED_SUCCESS = {
    "delayed_grounding": (False, True),
    "replan_after_miss": (False, True),
    "dispatch_committed_then_error": (False, True),
    "dispatch_unresolved": (False, False),
    "post_action_error": (False, True),
    "invalid_replan": (False, False),
    "step_budget_exhausted": (False, False),
    "replan_limit_exhausted": (False, False),
    "planner_timeout": (False, False),
}


def action_type_text(text: str) -> GUIAction:
    return GUIAction(action_type=ActionType.TYPE_TEXT, text=text)


def action_click(element: str) -> GUIAction:
    return GUIAction(action_type=ActionType.CLICK, element=element)


def plan_for(*items: tuple[str, GUIAction]) -> TaskPlan:
    return TaskPlan(steps=tuple(
        PlanStep(step_id=i, description=description, action=action)
        for i, (description, action) in enumerate(items, start=1)
    ))


@dataclass
class World:
    tokens: list[str] = field(default_factory=list)
    dispatched: list[str] = field(default_factory=list)
    attempted: list[str] = field(default_factory=list)
    observations: int = 0
    judge_calls: int = 0

    def observe(self, *, use_ocr: bool = False):
        self.observations += 1
        # Fake BGR screenshot; deliberately contains no real desktop content.
        return SimpleNamespace(
            screenshot=np.zeros((12, 16, 3), dtype=np.uint8),
            ocr_result=[] if use_ocr else None,
        )

    def judge(self, goal: tuple[str, ...]):
        self.judge_calls += 1
        ok = all(value in self.tokens for value in goal)
        return SuccessJudgeResult(
            success=ok,
            reason="synthetic goal met" if ok else "synthetic goal not met",
        )


def _build_synthetic_transaction(world: World, scenario: str):
    environment = SimpleNamespace(observe=world.observe)
    transaction = SimpleNamespace(environment=environment, last_failure=None)

    def execute(action: GUIAction):
        key = action.text if action.action_type == ActionType.TYPE_TEXT else action.element
        world.attempted.append(str(key))
        transaction.last_failure = None
        if action.action_type == ActionType.CLICK and action.element in {
            "Missing", "AlternateMissing"
        }:
            exc = GroundingTargetNotFoundError("synthetic target missing")
            transaction.last_failure = decide_recovery(exc, phase=FailurePhase.GROUNDING)
            raise exc
        if scenario in {"dispatch_committed_then_error", "post_action_error"} and key == "commit":
            world.tokens.append("commit")
            world.dispatched.append("commit")
            phase = (FailurePhase.POST_ACTION if scenario == "post_action_error"
                     else FailurePhase.ACTION_EXECUTION)
            exc = RuntimeError("synthetic uncertain action outcome")
            transaction.last_failure = decide_recovery(exc, phase=phase)
            raise exc
        if scenario == "dispatch_unresolved" and key == "commit":
            exc = RuntimeError("synthetic dispatch failed before visible effect")
            transaction.last_failure = decide_recovery(
                exc, phase=FailurePhase.ACTION_EXECUTION
            )
            raise exc
        if action.action_type == ActionType.TYPE_TEXT:
            world.tokens.append(str(action.text))
        world.dispatched.append(str(key))
        return SimpleNamespace(action=action, after=world.observe(use_ocr=False))

    transaction.execute = execute
    return transaction


def _delayed_grounding_case(world: World, mode: str):
    """Exercise the REAL ActionTransaction grounding retry loop with fake interfaces."""
    environment = Mock()
    environment.observe.side_effect = world.observe
    runner = Mock()
    runner.preflight.return_value = ActionRunRequirements(needs_ocr=True)
    action = action_click("Delayed")
    attempts = 0

    def ground_and_dispatch(act, before):
        nonlocal attempts
        attempts += 1
        world.attempted.append("Delayed")
        if attempts == 1:
            raise GroundingTargetNotFoundError("synthetic delayed target")
        world.tokens.append("clicked")
        world.dispatched.append("clicked")
        return act

    runner.run.side_effect = ground_and_dispatch
    feedback = Mock()
    feedback.compare.return_value = SimpleNamespace(changed=True)
    transaction = ActionTransaction(
        environment=environment,
        runner=runner,
        feedback=feedback,
        settle_seconds=0,
        retry_policy=RetryPolicy(
            max_grounding_attempts=2 if mode == "enhanced" else 1,
            grounding_retry_delay_s=0,
        ),
    )
    runtime = AgentRuntime.create(task="synthetic delayed target", max_steps=2, timeout_s=30)
    loop = AgentLoop(runtime=runtime, transaction=transaction)
    plan = plan_for(("Click delayed text", action))
    result = loop.run(plan, final_check=lambda: world.judge(("clicked",)))
    return result, runtime, 0


def run_trial(scenario: str, mode: str, iteration: int = 1) -> dict:
    """A single paired deterministic trial; real desktop is NEVER touched."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration must be a positive integer")

    world = World()
    goal: tuple[str, ...]
    start = time.perf_counter()

    if scenario == "delayed_grounding":
        result, runtime, replans = _delayed_grounding_case(world, mode)
    else:
        transaction = _build_synthetic_transaction(world, scenario)
        environment = transaction.environment
        planner = Mock()
        runtime = AgentRuntime.create(
            task=f"synthetic {scenario}",
            max_steps=2 if scenario == "step_budget_exhausted" else 5,
            timeout_s=30.0,
        )

        if scenario in {
            "replan_after_miss", "invalid_replan", "step_budget_exhausted",
            "replan_limit_exhausted", "planner_timeout",
        }:
            original = plan_for(
                ("Type prefix", action_type_text("prefix")),
                ("Click missing control", action_click("Missing")),
                ("Type done", action_type_text("done")),
            )
            goal = ("prefix", "done")
        else:
            original = plan_for(("Commit action", action_type_text("commit")))
            goal = ("commit",)

        replacement = {
            "invalid_replan": plan_for(("Repeat bad click", action_click("Missing"))),
            "replan_limit_exhausted": plan_for(
                ("Click other unavailable control", action_click("AlternateMissing"))
            ),
        }.get(scenario, plan_for(("Finish remaining task", action_type_text("done"))))

        def new_plan(instruction, screenshot_path=None):
            if screenshot_path is None or not Path(screenshot_path).is_file():
                raise AssertionError("Replan must receive a temporary screenshot")
            if scenario == "planner_timeout":
                # Simulated elapsed time; does not block the real test runner.
                runtime.state.started_at -= 100.0
            return replacement

        planner.plan.side_effect = new_plan
        recovery = RecoveryEngine(environment) if mode == "enhanced" else None
        replanner = Replanner(planner, max_replans=1) if mode == "enhanced" else None
        loop = AgentLoop(
            runtime=runtime,
            transaction=transaction,
            recovery_engine=recovery,
            replanner=replanner,
        )
        result = loop.run(
            original,
            final_check=lambda: world.judge(goal),
            recovery_check=(lambda observation: world.judge(goal)) if recovery else None,
        )
        replans = planner.plan.call_count

    elapsed_s = time.perf_counter() - start
    expected = EXPECTED_SUCCESS[scenario][1 if mode == "enhanced" else 0]
    repeated_dispatch = any(count > 1 for count in Counter(world.dispatched).values())
    # Critical invariant: uncertain actions must be ATTEMPTED once, not replayed.
    unsafe_retry = scenario in {
        "dispatch_committed_then_error", "dispatch_unresolved", "post_action_error"
    } and world.attempted.count("commit") != 1
    passed = result.success is expected and not repeated_dispatch and not unsafe_retry
    if scenario == "step_budget_exhausted" and mode == "enhanced":
        passed &= replans == 0
    if scenario == "replan_limit_exhausted" and mode == "enhanced":
        passed &= replans == 1
    if scenario == "planner_timeout" and mode == "enhanced":
        passed &= runtime.state.done is False

    recovery_status = (
        result.recovery_result.status.value if result.recovery_result else "none"
    )
    decision_phase = (
        result.failure_decision.phase.value if result.failure_decision else "none"
    )
    return {
        "dataset": "synthetic_fault_injection",
        "scenario": scenario,
        "mode": mode,
        "iteration": iteration,
        "auto_success": bool(result.success),
        "expected_success": bool(expected),
        "safety_pass": bool(passed),
        "duration_s": round(elapsed_s, 6),
        "step_count": runtime.state.step_count,
        "attempt_count": len(world.attempted),
        "dispatched_count": len(world.dispatched),
        "fresh_observations": world.observations,
        "replan_calls": replans,
        "judge_calls": world.judge_calls,
        "failure_phase": decision_phase,
        "recovery_status": recovery_status,
        "failure_type": "none" if result.success else (
            result.failure_decision.exception_type
            if result.failure_decision else "unclassified"
        ),
        # Intentionally do not store raw OCR, screenshots, prompts or exception texts.
    }


def percentile_95(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("no data")
    position = (len(values) - 1) * 0.95
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario"], record["mode"])].append(record)
    rows = []
    for (scenario, mode), data in sorted(groups.items()):
        durations = [item["duration_s"] for item in data]
        rows.append({
            "scenario": scenario,
            "mode": mode,
            "trials": len(data),
            "successes": sum(item["auto_success"] for item in data),
            "success_rate": round(statistics.mean(item["auto_success"] for item in data), 4),
            "safety_pass_rate": round(statistics.mean(item["safety_pass"] for item in data), 4),
            "mean_s": round(statistics.mean(durations), 6),
            "p95_s": round(percentile_95(durations), 6),
            "std_s_population": round(statistics.pstdev(durations), 6),
            "mean_steps": round(statistics.mean(item["step_count"] for item in data), 3),
            "mean_replans": round(statistics.mean(item["replan_calls"] for item in data), 3),
        })
    return rows


def write_reports(records: list[dict], out_dir: Path) -> list[dict]:
    if not records:
        raise ValueError("cannot report empty results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "synthetic_raw.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    with (out_dir / "synthetic_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "synthetic_report.json").write_text(
        json.dumps({
            "kind": "simulation_not_real_desktop",
            "caveat": (
                "No physical mouse, OCR, model inference, or external app was used. "
                "Timings are mock execution times, NOT production latency."
            ),
            "raw_trials": len(records),
            "all_safety_checks_passed": all(r["safety_pass"] for r in records),
            "summary": summary,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="Trials per scenario/mode (default 10)")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/robustness/week6"))
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")
    cases = SCENARIOS if args.scenario == "all" else (args.scenario,)
    records = [
        run_trial(scenario, mode, i)
        for i in range(1, args.runs + 1)
        for scenario in cases
        for mode in MODES
    ]
    summary = write_reports(records, args.out_dir)
    print("MOCK-ONLY FAILURE INJECTION; NOT REAL GUI PERFORMANCE")
    for row in summary:
        print(
            f"{row['scenario']:<31} {row['mode']:<8} "
            f"{row['successes']:>2}/{row['trials']:<2} "
            f"safety={row['safety_pass_rate']:.0%} "
            f"mean={row['mean_s']:.6f}s"
        )
    print("Reports:", args.out_dir.resolve())
    return 0 if all(row["safety_pass_rate"] == 1 for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
