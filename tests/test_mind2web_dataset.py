import json

import pytest

from gui_agent.datasets.mind2web import (
    load_mind2web_file,
    normalize_mind2web_candidate,
    parse_mind2web_action,
    parse_mind2web_task,
)
from gui_agent.datasets.schema import ActionType


def candidate(
    *,
    node_id="123",
    original=True,
    top_level=True,
    bbox="10,20,30,40",
):
    attrs = {
        "backend_node_id": node_id,
        "aria_label": "Search",
    }
    if bbox is not None:
        attrs["bounding_box_rect"] = bbox

    return {
        "tag": "button",
        "attributes": json.dumps(attrs),
        "is_original_target": original,
        "is_top_level_target": top_level,
        "backend_node_id": node_id,
    }


def action(
    op="CLICK",
    *,
    value="",
    pos_candidates=None,
    neg_candidates=None,
    uid="action_001",
    original_op=None,
):
    if pos_candidates is None:
        pos_candidates = [candidate()]
    if neg_candidates is None:
        neg_candidates = []

    return {
        "action_uid": uid,
        "raw_html": "<html>raw</html>",
        "cleaned_html": "<html>cleaned</html>",
        "operation": {
            "original_op": original_op or op,
            "value": value,
            "op": op,
        },
        "pos_candidates": pos_candidates,
        "neg_candidates": neg_candidates,
    }


def task(actions):
    return {
        "website": "example",
        "domain": "Shopping",
        "subdomain": "Retail",
        "annotation_id": "task_001",
        "confirmed_task": "Search for GUI Agent",
        "action_reprs": [f"step-{i}" for i in range(len(actions))],
        "actions": actions,
    }


def test_candidate_attributes_and_bbox_are_normalized():
    normalized = normalize_mind2web_candidate(candidate())

    assert normalized["tag"] == "button"
    assert normalized["backend_node_id"] == "123"
    assert normalized["bbox"] == {
        "x": 10.0,
        "y": 20.0,
        "width": 30.0,
        "height": 40.0,
    }
    assert normalized["labels"]["aria_label"] == "Search"
    assert normalized["attributes"]["backend_node_id"] == "123"


def test_candidate_without_bbox_is_valid():
    normalized = normalize_mind2web_candidate(
        candidate(bbox=None)
    )
    assert normalized["bbox"] is None


def test_invalid_candidate_attributes_are_rejected():
    raw = candidate()
    raw["attributes"] = "{not-json"

    with pytest.raises(ValueError, match="valid JSON"):
        normalize_mind2web_candidate(raw)


def test_click_action_keeps_dom_target_out_of_desktop_position():
    parsed = parse_mind2web_action(action("CLICK"))

    assert parsed.action_type == ActionType.CLICK
    assert parsed.position is None
    assert parsed.metadata["positive_candidate_count"] == 1
    assert parsed.metadata["original_target_count"] == 1


def test_empty_positive_candidates_are_valid():
    parsed = parse_mind2web_action(
        action(
            "CLICK",
            pos_candidates=[],
            neg_candidates=[candidate(original=False)],
        )
    )

    assert parsed.action_type == ActionType.CLICK
    assert parsed.metadata["positive_candidate_count"] == 0
    assert (
        parsed.metadata["candidate_status"]
        == "no_positive_candidate_in_cleaned_html"
    )




def test_negative_candidate_target_flags_are_not_required_booleans():
    negative = candidate(node_id="neg-1", original=False)
    negative["is_original_target"] = None
    negative["is_top_level_target"] = None

    parsed = parse_mind2web_action(
        action(
            "CLICK",
            neg_candidates=[negative],
        )
    )

    assert parsed.metadata["negative_candidate_count"] == 1


def test_negative_candidate_requires_backend_node_id():
    negative = candidate(node_id="neg-1", original=False)
    negative.pop("backend_node_id")

    with pytest.raises(ValueError, match="backend_node_id"):
        parse_mind2web_action(
            action(
                "CLICK",
                neg_candidates=[negative],
            )
        )


def test_negative_candidate_must_be_object():
    with pytest.raises(ValueError, match="candidate must be an object"):
        parse_mind2web_action(
            action(
                "CLICK",
                neg_candidates=["bad-candidate"],
            )
        )


def test_multiple_positive_candidates_are_preserved():
    parsed = parse_mind2web_action(
        action(
            "CLICK",
            pos_candidates=[
                candidate(node_id="1", original=True),
                candidate(node_id="2", original=False),
            ],
        )
    )

    assert parsed.metadata["positive_candidate_count"] == 2
    assert len(parsed.metadata["positive_candidates"]) == 2
    assert parsed.metadata["original_target_count"] == 1


def test_multiple_original_targets_are_preserved_not_collapsed():
    parsed = parse_mind2web_action(
        action(
            "CLICK",
            pos_candidates=[
                candidate(node_id="1", original=True),
                candidate(node_id="2", original=True),
            ],
        )
    )

    assert parsed.metadata["original_target_count"] == 2
    assert len(parsed.metadata["original_target_candidates"]) == 2


def test_type_action():
    parsed = parse_mind2web_action(
        action("TYPE", value="GUI Agent")
    )

    assert parsed.action_type == ActionType.TYPE_TEXT
    assert parsed.text == "GUI Agent"


def test_select_action():
    parsed = parse_mind2web_action(
        action("SELECT", value="Option A")
    )

    assert parsed.action_type == ActionType.SELECT
    assert parsed.text == "Option A"


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_mind2web_action(action("UNKNOWN"))


def test_task_conversion_uses_annotation_id_and_action_repr():
    data = task(
        [
            action("CLICK", uid="a1"),
            action("TYPE", value="GUI Agent", uid="a2"),
        ]
    )

    examples = parse_mind2web_task(data, source_file="train_0.json")

    assert len(examples) == 2
    assert examples[0].task_id == "task_001"
    assert examples[0].metadata["source_file"] == "train_0.json"
    assert examples[0].metadata["action_repr"] == "step-0"
    assert examples[1].step_index == 1
    assert len(examples[1].history) == 1


def test_action_repr_length_must_match_actions():
    data = task([action("CLICK")])
    data["action_reprs"] = []

    with pytest.raises(ValueError, match="length"):
        parse_mind2web_task(data)


def test_load_file(tmp_path):
    path = tmp_path / "train_0.json"
    path.write_text(
        json.dumps([task([action("CLICK")])]),
        encoding="utf-8",
    )

    examples = load_mind2web_file(path)

    assert len(examples) == 1
    assert examples[0].task_id == "task_001"
    assert examples[0].metadata["source_file"] == "train_0.json"
