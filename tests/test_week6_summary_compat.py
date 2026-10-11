"""Compatibility of real desktop A/B summary with pre-ocr-device fixtures."""

from scripts.robustness.week6_real_summary import compare_modes, summarize


def _group(mode: str, *, fault: str = "none", ocr_device=None):
    group = {
        "task": "browser_search",
        "fault": fault,
        "mode": mode,
        "replan_backend": "none",
        "runs": 10,
        "auto_success_rate": 0.8 if mode == "enhanced" else 0.5,
        "mean_duration_s": 8.5 if mode == "enhanced" else 7.0,
    }
    if ocr_device is not None:
        group["ocr_device"] = ocr_device
    return group


def test_legacy_groups_pair_as_gpu_without_ocr_device():
    comparison = compare_modes([_group("baseline"), _group("enhanced")])
    assert len(comparison) == 1
    assert comparison[0]["ocr_device"] == "gpu:0"
    assert comparison[0]["auto_success_gain_pp"] == 30.0


def test_legacy_gpu_group_does_not_pair_with_cpu():
    comparison = compare_modes([
        _group("baseline"), _group("enhanced", ocr_device="cpu")
    ])
    assert comparison == []


def test_group_comparison_only_pairs_matching_fault():
    comparison = compare_modes([
        _group("baseline"),
        _group("enhanced"),
        _group("enhanced", fault="grounding_once"),
    ])
    assert len(comparison) == 1
    assert comparison[0]["fault"] == "none"


def test_raw_legacy_real_trials_default_gpu_and_group_separately():
    base = {
        "source": "real_desktop_controlled", "task_base": "text_entry",
        "fault": "none", "mode": "baseline", "replan_backend": "none",
        "duration_s": 12.0, "auto_success": True,
        "human_success": True, "steps": 4,
    }
    groups = summarize([base, {**base, "ocr_device": "cpu"}])
    assert {g["ocr_device"] for g in groups} == {"cpu", "gpu:0"}
