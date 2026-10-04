import ctypes

import pytest

import gui_agent.control.windows_text as windows_text


class FakeSendInput:
    def __init__(self):
        self.calls = []
        self.return_value = None

    def __call__(
        self,
        count,
        inputs,
        input_size,
    ):
        events = []

        for index in range(count):
            item = inputs[index]

            events.append(
                {
                    "type": item.type,
                    "wVk": item.ki.wVk,
                    "wScan": item.ki.wScan,
                    "dwFlags": item.ki.dwFlags,
                }
            )

        self.calls.append(
            {
                "count": count,
                "input_size": input_size,
                "events": events,
            }
        )

        if self.return_value is not None:
            return self.return_value

        return count


@pytest.fixture
def fake_send_input(monkeypatch):
    fake = FakeSendInput()

    monkeypatch.setattr(
        windows_text,
        "SendInput",
        fake,
    )

    return fake


def test_send_utf16_unit_sends_key_down_and_key_up(
    fake_send_input,
):
    windows_text._send_utf16_unit(
        0x0041
    )

    assert len(
        fake_send_input.calls
    ) == 1

    call = fake_send_input.calls[0]

    assert call["count"] == 2
    assert call["input_size"] == ctypes.sizeof(
        windows_text.INPUT
    )

    key_down = call["events"][0]
    key_up = call["events"][1]

    assert key_down == {
        "type": windows_text.INPUT_KEYBOARD,
        "wVk": 0,
        "wScan": 0x0041,
        "dwFlags": windows_text.KEYEVENTF_UNICODE,
    }

    assert key_up == {
        "type": windows_text.INPUT_KEYBOARD,
        "wVk": 0,
        "wScan": 0x0041,
        "dwFlags": (
            windows_text.KEYEVENTF_UNICODE
            | windows_text.KEYEVENTF_KEYUP
        ),
    }


@pytest.mark.parametrize(
    ("text", "expected_units"),
    [
        (
            "A",
            [0x0041],
        ),
        (
            "中",
            [0x4E2D],
        ),
        (
            "A中",
            [
                0x0041,
                0x4E2D,
            ],
        ),
        (
            "é",
            [0x00E9],
        ),
        (
            "🙂",
            [
                0xD83D,
                0xDE42,
            ],
        ),
        (
            "",
            [],
        ),
    ],
)
def test_type_unicode_text_encodes_utf16_units(
    monkeypatch,
    text,
    expected_units,
):
    sent_units = []

    monkeypatch.setattr(
        windows_text,
        "_send_utf16_unit",
        sent_units.append,
    )

    windows_text.type_unicode_text(
        text
    )

    assert sent_units == expected_units


def test_type_unicode_text_uses_interval_per_character(
    monkeypatch,
):
    sent_units = []
    sleep_calls = []

    monkeypatch.setattr(
        windows_text,
        "_send_utf16_unit",
        sent_units.append,
    )

    monkeypatch.setattr(
        windows_text.time,
        "sleep",
        sleep_calls.append,
    )

    windows_text.type_unicode_text(
        "AB",
        interval=0.03,
    )

    assert sent_units == [
        0x0041,
        0x0042,
    ]

    assert sleep_calls == [
        0.03,
        0.03,
    ]


def test_emoji_interval_occurs_after_complete_surrogate_pair(
    monkeypatch,
):
    operations = []

    def fake_send(code_unit):
        operations.append(
            ("send", code_unit)
        )

    def fake_sleep(interval):
        operations.append(
            ("sleep", interval)
        )

    monkeypatch.setattr(
        windows_text,
        "_send_utf16_unit",
        fake_send,
    )

    monkeypatch.setattr(
        windows_text.time,
        "sleep",
        fake_sleep,
    )

    windows_text.type_unicode_text(
        "🙂",
        interval=0.01,
    )

    assert operations == [
        (
            "send",
            0xD83D,
        ),
        (
            "send",
            0xDE42,
        ),
        (
            "sleep",
            0.01,
        ),
    ]


def test_empty_text_sends_nothing(
    monkeypatch,
):
    sent_units = []

    monkeypatch.setattr(
        windows_text,
        "_send_utf16_unit",
        sent_units.append,
    )

    windows_text.type_unicode_text(
        ""
    )

    assert sent_units == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        1,
        1.5,
        [],
        {},
    ],
)
def test_type_unicode_text_requires_string(
    bad,
):
    with pytest.raises(TypeError):
        windows_text.type_unicode_text(
            bad
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "interval",
    [
        -0.01,
        -1,
    ],
)
def test_type_unicode_text_rejects_negative_interval(
    interval,
):
    with pytest.raises(
        ValueError,
        match="interval",
    ):
        windows_text.type_unicode_text(
            "x",
            interval=interval,
        )


def test_send_utf16_unit_raises_when_sendinput_is_incomplete(
    fake_send_input,
    monkeypatch,
):
    fake_send_input.return_value = 1

    monkeypatch.setattr(
        windows_text.ctypes,
        "get_last_error",
        lambda: 5,
    )

    with pytest.raises(
        RuntimeError,
        match="SendInput sent 1/2",
    ) as exc_info:
        windows_text._send_utf16_unit(
            0x0041
        )

    message = str(
        exc_info.value
    )

    assert (
        "Windows error code: 5"
        in message
    )

    assert (
        "UIPI"
        in message
    )


def test_send_utf16_unit_accepts_successful_send(
    fake_send_input,
):
    fake_send_input.return_value = 2

    windows_text._send_utf16_unit(
        0x4E2D
    )

    assert len(
        fake_send_input.calls
    ) == 1