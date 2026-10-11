"""No desktop/GPU use. Validate the safe diagnostics and distinct fault modes."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'robustness'))
from week6_real_e2e import TrialSpec, safe_failure_code, should_inject_grounding

@pytest.mark.parametrize('fault,expected', [
    ('grounding_once', [True, False, False]),
    ('grounding_exhausted', [True, True, True]),
    ('grounding_replan_window', [True, True, False]),
    ('none', [False, False, False]),
])
def test_grounding_fault_scopes(fault, expected):
    assert [should_inject_grounding(fault, i) for i in range(3)] == expected

@pytest.mark.parametrize('reason,expected', [
    ('injected pre-dispatch target miss; replanning failed: ReplanError: new plan repeats the failed action', 'replan_repeats_failed_action'),
    ('injected pre-dispatch target miss; replanning failed: PlannerOutputError: invalid schema', 'planner_output_invalid'),
    ('timeout exceeded: 90.00s', 'timeout'),
    ('injected pre-dispatch target miss; replanning failed: RuntimeError: private screenshot text', 'replan_RuntimeError'),
    (None, 'none'),
])
def test_safe_failure_code(reason, expected):
    assert safe_failure_code(reason) == expected
    if reason and 'private' in reason:
        assert 'private' not in safe_failure_code(reason)


def test_new_fault_allowed_and_browser_rejected():
    assert TrialSpec('text_entry', 'enhanced', 2, 'grounding_replan_window', 'local-vlm').fault == 'grounding_replan_window'
    with pytest.raises(ValueError):
        TrialSpec('browser_open', 'enhanced', 2, 'grounding_replan_window', 'local-vlm')
