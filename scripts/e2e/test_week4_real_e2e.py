from gui_agent.agent.action_runner import ActionRunner
from gui_agent.agent.action_transaction import ActionTransaction
from gui_agent.agent.executor import ActionExecutor
from gui_agent.agent.loop import AgentLoop
from gui_agent.agent.plan_schema import TaskPlan
from gui_agent.agent.runtime import AgentRuntime
from gui_agent.agent.environment import DesktopEnvironment
from gui_agent.control.controller import Controller
from gui_agent.agent.success_judge import TextSuccessJudge
from gui_agent.ocr.paddleocr_engine import PaddleOCREngine



def main():
    plan = TaskPlan.from_dict(
        {
            "steps": [
                {
                    "step_id": 1,
                    "description": "Open Windows search",
                    "action": {
                        "action_type": "key_press",
                        "keys": ["win"],
                    },
                },
                {
                    "step_id": 2,
                    "description": "Type Notepad",
                    "action": {
                        "action_type": "type_text",
                        "text": "Notepad",
                    },
                },
                {
                    "step_id": 3,
                    "description": "Launch Notepad",
                    "action": {
                        "action_type": "key_press",
                        "keys": ["enter"],
                    },
                },
                {
                    "step_id": 4,
                    "description": "Wait for Notepad",
                    "action": {
                        "action_type": "wait",
                        "wait_seconds": 2.0,
                    },
                },
                {
                    "step_id": 5,
                    "description": "Type test text",
                    "action": {
                        "action_type": "type_text",
                        "text": "Week4 GUI Agent E2E Test",
                    },
                },
            ]
        }
    )

    controller = Controller()
    ocr_engine = PaddleOCREngine(
        lang="ch",
        device="gpu:0",
        min_confidence=0.5,
    )

    environment = DesktopEnvironment(
        ocr_engine=ocr_engine,
    )

    executor = ActionExecutor(controller)
    runner = ActionRunner(executor)

    transaction = ActionTransaction(
        environment,
        runner,
        settle_seconds=0.5,
    )

    runtime = AgentRuntime.create(
        task="Open Firefox",
        max_steps=6,
        timeout_s=30.0,
    )

    loop = AgentLoop(
        runtime,
        transaction,
    )

    try:
        judge = TextSuccessJudge()

        def final_check():
            observation = environment.observe(
                use_ocr=True
            )

            return judge.judge(
                observation,
                "Week4 GUI Agent E2E Test",
            )

        result = loop.run(
            plan,
            final_check=final_check,
        )

        print("\n=== Week 4 Real E2E ===")
        print("success:", result.success)
        print("steps:", runtime.state.step_count)
        print("done:", runtime.state.done)
        print("failed:", runtime.state.failed)

        if result.judge_result is not None:
            print(
                "judge_success:",
                result.judge_result.success,
            )
            print(
                "judge_reason:",
                result.judge_result.reason,
            )

        if result.failure_reason:
            print(
                "failure_reason:",
                result.failure_reason,
            )

        for index, tx in enumerate(
            result.transactions,
            start=1,
        ):
            print(
                f"step {index}: "
                f"action={tx.action.action_type.value}, "
                f"changed={tx.feedback.changed}, "
                f"ratio={tx.feedback.changed_pixel_ratio:.6f}, "
                f"mean_diff={tx.feedback.mean_abs_diff:.4f}"
            )

        if result.failure_reason:
            print(
                "failure_reason:",
                result.failure_reason,
            )

    finally:
        environment.close()


if __name__ == "__main__":
    main()