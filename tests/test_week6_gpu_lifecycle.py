"""No-GPU tests for one-shot VLM cleanup and CPU OCR A/B isolation."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from gui_agent.agent.plan_schema import PlanStep, TaskPlan
from gui_agent.agent.replan import ReplanContext, Replanner, ReplanError
from gui_agent.datasets.schema import GUIAction, ActionType
from scripts.robustness import week6_real_e2e as e2e
from scripts.robustness import week6_real_summary as summary


def context():
    return ReplanContext(
        task="type a message", completed_steps=(),
        failed_step=PlanStep(1, "click target", GUIAction(action_type=ActionType.CLICK, element="E2E_TARGET")),
        pending_steps=(), failure_reason="missing text target", attempt=1, available_steps=2,
    )


def valid_plan():
    return TaskPlan(steps=(PlanStep(1, "type text", GUIAction(action_type=ActionType.TYPE_TEXT, text="hello")),))


def fake_planner(*, response=None, exception=None, with_model=True):
    class FakePlanner:
        def __init__(self):
            self.model = Mock() if with_model else None
        def plan(self, instruction, screenshot_path=None):
            if exception is not None:
                raise exception
            return response
    return FakePlanner()


def test_model_unloaded_after_success():
    planner = fake_planner(response=valid_plan())
    result = Replanner(planner).replan(context())
    assert result == valid_plan()
    planner.model.unload.assert_called_once_with()


def test_model_unloaded_after_inference_exception():
    planner = fake_planner(exception=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        Replanner(planner).replan(context())
    planner.model.unload.assert_called_once_with()


def test_model_unloaded_when_output_is_invalid():
    planner = fake_planner(response={"steps": []})
    with pytest.raises(ReplanError, match="TaskPlan"):
        Replanner(planner).replan(context())
    planner.model.unload.assert_called_once_with()


def test_no_unload_method_remains_supported():
    planner = fake_planner(response=valid_plan(), with_model=False)
    assert Replanner(planner).replan(context()) == valid_plan()


def test_invalid_context_does_not_call_model():
    planner = fake_planner(response=valid_plan())
    with pytest.raises(TypeError):
        Replanner(planner).replan(object())
    planner.model.unload.assert_not_called()


def test_cli_cpu_ocr_is_forwarded_without_desktop(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(e2e, "run_trial", lambda *a, **kw: recorded.append(kw))
    assert e2e.main(["--task", "text_entry", "--mode", "enhanced", "--trial", "5",
                     "--replan-backend", "local-vlm", "--ocr-device", "cpu",
                     "--output", str(tmp_path), "--execute"]) == 0
    assert recorded[0]["ocr_device"] == "cpu"


def test_cli_default_ocr_unchanged(monkeypatch, tmp_path):
    recorded = []
    monkeypatch.setattr(e2e, "run_trial", lambda *a, **kw: recorded.append(kw))
    assert e2e.main(["--task", "text_entry", "--mode", "enhanced", "--trial", "5",
                     "--output", str(tmp_path), "--execute"]) == 0
    assert recorded[0]["ocr_device"] == "gpu:0"


def row(mode, device, trial):
    return {"source":"real_desktop_controlled", "task_base":"text_entry",
            "fault":"grounding_replan_window", "mode": mode,
            "trial":trial, "replan_backend":"none", "ocr_device":device,
            "duration_s":1.0, "human_success":False, "auto_success":False, "steps":1}


def test_summary_separates_cpu_from_gpu():
    groups = summary.summarize([row("baseline", "gpu:0", 1), row("baseline", "cpu", 2)])
    assert len(groups) == 2
    assert {g["ocr_device"] for g in groups} == {"cpu", "gpu:0"}


def test_summary_does_not_pair_different_ocr_devices():
    groups = summary.summarize([row("baseline", "gpu:0", 1), row("enhanced", "cpu", 1)])
    assert summary.compare_modes(groups) == []


def test_legacy_records_default_to_gpu():
    prior = row("baseline", "gpu:0", 1)
    prior.pop("ocr_device")
    assert summary.summarize([prior])[0]["ocr_device"] == "gpu:0"
