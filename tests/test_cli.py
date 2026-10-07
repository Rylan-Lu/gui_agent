from __future__ import annotations

import pytest

from gui_agent.__main__ import build_parser


def test_cli_defaults_to_local_backend() -> None:
    args = build_parser().parse_args(
        ["--task", "Open Notepad"]
    )

    assert args.task == "Open Notepad"
    assert args.backend == "local"
    assert args.model is None
    assert args.max_new_tokens == 384
    assert args.max_steps == 10
    assert args.timeout == 60.0


def test_cli_supports_api_configuration() -> None:
    args = build_parser().parse_args(
        [
            "--task",
            "Search for von Neumann",
            "--backend",
            "api",
            "--endpoint",
            "https://example.com/v1/chat/completions",
            "--model",
            "vision-model",
        ]
    )

    assert args.backend == "api"
    assert args.endpoint.endswith("/chat/completions")
    assert args.model == "vision-model"


def test_cli_verification_options_are_mutually_exclusive() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--task",
                "Do task",
                "--expect-text",
                "DONE",
                "--expect-absent",
                "WAITING",
            ]
        )
