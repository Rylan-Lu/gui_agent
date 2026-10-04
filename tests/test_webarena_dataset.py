import json

import pytest

from gui_agent.datasets.webarena import (
    WebArenaTask,
    load_webarena_collection,
    load_webarena_file,
    parse_webarena_task,
)


def make_task(task_id=2):
    return {
        "sites": ["shopping_admin"],
        "task_id": task_id,
        "require_login": True,
        "storage_state": "./.auth/shopping_admin_state.json",
        "start_url": "__SHOPPING_ADMIN__",
        "geolocation": None,
        "intent_template": "What is the top-1 {{metric}}",
        "instantiation_dict": {"metric": "product"},
        "intent": "What is the top-1 product",
        "require_reset": False,
        "eval": {
            "eval_types": ["string_match"],
            "reference_answers": {"exact_match": "Widget"},
            "reference_url": "",
            "program_html": [],
            "string_note": "",
            "reference_answer_raw_annotation": "Widget",
        },
        "intent_template_id": 279,
    }


def test_parse_task():
    task = parse_webarena_task(make_task())

    assert isinstance(task, WebArenaTask)
    assert task.task_id == 2
    assert task.sites == ("shopping_admin",)
    assert task.start_url == "__SHOPPING_ADMIN__"
    assert task.intent_template_id == 279
    assert task.eval_types == ("string_match",)
    assert task.reference_action_sequence is None
    assert task.reference_action_count == 0


def test_optional_reference_actions_are_still_supported():
    raw = make_task()
    raw["reference_action_sequence"] = {
        "action_set_tag": "playwright",
        "action_sequence": ["click", "stop"],
    }

    task = parse_webarena_task(raw)

    assert task.reference_action_count == 2


def test_program_html_payload_is_preserved():
    raw = make_task()
    raw["eval"] = {
        "eval_types": ["program_html"],
        "reference_answers": None,
        "reference_url": None,
        "program_html": [
            {
                "url": "last",
                "locator": "",
                "required_contents": {
                    "must_include": ["jaw", "bruxism"],
                },
            }
        ],
    }

    task = parse_webarena_task(raw)

    assert task.evaluation["program_html"][0]["url"] == "last"
    assert task.evaluation["reference_url"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", True),
        ("task_id", "2"),
        ("task_id", -1),
        ("intent", ""),
        ("sites", []),
        ("start_url", ""),
        ("require_login", 1),
        ("require_reset", 0),
        ("storage_state", 123),
        ("geolocation", "Pittsburgh"),
        ("intent_template", None),
        ("intent_template_id", True),
        ("instantiation_dict", []),
    ],
)
def test_invalid_top_level_fields_rejected(field, value):
    raw = make_task()
    raw[field] = value

    with pytest.raises(ValueError):
        parse_webarena_task(raw)


def test_invalid_evaluation_rejected():
    raw = make_task()
    raw["eval"]["eval_types"] = "string_match"

    with pytest.raises(ValueError, match="eval_types"):
        parse_webarena_task(raw)


def test_missing_required_eval_key_rejected():
    raw = make_task()
    del raw["eval"]["program_html"]

    with pytest.raises(ValueError, match="missing required"):
        parse_webarena_task(raw)


def test_invalid_program_html_item_rejected():
    raw = make_task()
    raw["eval"]["program_html"] = ["bad"]

    with pytest.raises(ValueError, match="program_html"):
        parse_webarena_task(raw)


def test_load_single_file(tmp_path):
    path = tmp_path / "2.json"
    path.write_text(json.dumps(make_task()), encoding="utf-8")

    task = load_webarena_file(path)

    assert task.task_id == 2
    assert task.source_file == str(path)


def test_load_collection(tmp_path):
    path = tmp_path / "test.raw.json"
    path.write_text(
        json.dumps([make_task(0), make_task(1)]),
        encoding="utf-8",
    )

    tasks = load_webarena_collection(path)

    assert [task.task_id for task in tasks] == [0, 1]
    assert all(task.source_file == str(path) for task in tasks)


def test_single_loader_rejects_collection(tmp_path):
    path = tmp_path / "test.raw.json"
    path.write_text(json.dumps([make_task(0)]), encoding="utf-8")

    with pytest.raises(ValueError, match="collection"):
        load_webarena_file(path)
