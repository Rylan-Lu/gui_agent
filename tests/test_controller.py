import importlib
import sys
import types

import pytest


class FakePyAutoGUI(types.ModuleType):
    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE = False
        self.calls = []

    def size(self):
        return (2560, 1440)

    def moveTo(self, x, y, duration=0):
        self.calls.append(("moveTo", x, y, duration))

    def click(self, button="left"):
        self.calls.append(("click", button))

    def doubleClick(self, button="left", interval=0):
        self.calls.append(("doubleClick", button, interval))

    def dragTo(self, x, y, duration=0, button="left"):
        self.calls.append(("dragTo", x, y, duration, button))

    def write(self, text, interval=0):
        self.calls.append(("write", text, interval))

    def press(self, key, presses=1, interval=0):
        self.calls.append(("press", key, presses, interval))

    def hotkey(self, *keys, interval=0):
        self.calls.append(("hotkey", keys, interval))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))


fake_pyautogui = FakePyAutoGUI()
fake_windows_text = types.ModuleType("gui_agent.control.windows_text")
fake_windows_text.calls = []


def fake_type_unicode_text(text):
    fake_windows_text.calls.append(text)


fake_windows_text.type_unicode_text = fake_type_unicode_text
sys.modules["pyautogui"] = fake_pyautogui
sys.modules["gui_agent.control.windows_text"] = fake_windows_text
sys.modules.pop("gui_agent.control.controller", None)
controller_module = importlib.import_module("gui_agent.control.controller")
Controller = controller_module.Controller
ControlError = controller_module.ControlError


@pytest.fixture
def controller():
    fake_pyautogui.calls.clear()
    fake_windows_text.calls.clear()
    return Controller()


def test_controller_initialization(controller):
    assert controller.screen_size == (2560, 1440)
    assert fake_pyautogui.FAILSAFE is True


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0, 0), (0, 0)),
        ((1.2, 2.8), (1, 3)),
        ((100.5, 200.5), (100, 200)),
        ((2559, 1439), (2559, 1439)),
        ((1280.4, 720.6), (1280, 721)),
    ],
)
def test_round_point(point, expected):
    assert Controller._round_point(point) == expected


@pytest.mark.parametrize("point", [(0, 0), (1, 1), (2559, 1439), (1280, 720), (0.1, 0.1)])
def test_move_to_valid(controller, point):
    controller.move_to(point, duration=0.25)
    x, y = Controller._round_point(point)
    assert fake_pyautogui.calls[-1] == ("moveTo", x, y, 0.25)


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (2560, 0), (0, 1440), (3000, 2000), (-0.1, 1)])
def test_move_to_out_of_bounds(controller, point):
    with pytest.raises(ControlError):
        controller.move_to(point)


@pytest.mark.parametrize("duration", [-0.01, -1, -100])
def test_negative_duration_rejected(controller, duration):
    with pytest.raises(ValueError):
        controller.move_to((10, 10), duration=duration)


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_click(controller, button):
    controller.click((10, 20), button=button, duration=0.1)
    assert fake_pyautogui.calls == [("moveTo", 10, 20, 0.1), ("click", button)]


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_double_click(controller, button):
    controller.double_click((10, 20), button=button, duration=0.1, interval=0.2)
    assert fake_pyautogui.calls == [
        ("moveTo", 10, 20, 0.1),
        ("doubleClick", button, 0.2),
    ]


@pytest.mark.parametrize("button", ["left", "right", "middle"])
def test_drag_to(controller, button):
    controller.drag_to((10.4, 20.6), button=button, duration=0.7)
    assert fake_pyautogui.calls == [("dragTo", 10, 21, 0.7, button)]


@pytest.mark.parametrize("text", ["hello", "123", "A B C", "", "file_name.txt"])
def test_type_ascii_uses_pyautogui(controller, text):
    controller.type_text(text, interval=0.03)
    assert fake_pyautogui.calls == [("write", text, 0.03)]
    assert fake_windows_text.calls == []


@pytest.mark.parametrize("text", ["中文", "测试ABC", "é", "🙂"])
def test_type_unicode_uses_windows_input(controller, text):
    controller.type_text(text)
    assert fake_windows_text.calls == [text]
    assert fake_pyautogui.calls == []


@pytest.mark.parametrize("bad", [None, 1, 1.2, [], {}])
def test_type_text_requires_string(controller, bad):
    with pytest.raises(TypeError):
        controller.type_text(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("interval", [-0.1, -1])
def test_type_text_rejects_negative_interval(controller, interval):
    with pytest.raises(ValueError):
        controller.type_text("x", interval=interval)


@pytest.mark.parametrize(("presses", "interval"), [(1, 0), (2, 0.1), (5, 0.2)])
def test_press(controller, presses, interval):
    controller.press("enter", presses=presses, interval=interval)
    assert fake_pyautogui.calls == [("press", "enter", presses, interval)]


@pytest.mark.parametrize("presses", [0, -1, -10])
def test_press_rejects_non_positive_count(controller, presses):
    with pytest.raises(ValueError):
        controller.press("enter", presses=presses)


def test_press_rejects_negative_interval(controller):
    with pytest.raises(ValueError):
        controller.press("enter", interval=-0.1)


@pytest.mark.parametrize("keys", [("ctrl", "c"), ("ctrl", "shift", "s"), ("alt", "f4")])
def test_hotkey(controller, keys):
    controller.hotkey(*keys, interval=0.25)
    assert fake_pyautogui.calls == [("hotkey", keys, 0.25)]


def test_hotkey_requires_key(controller):
    with pytest.raises(ValueError):
        controller.hotkey()


def test_hotkey_rejects_negative_interval(controller):
    with pytest.raises(ValueError):
        controller.hotkey("ctrl", "c", interval=-0.1)


@pytest.mark.parametrize("amount", [-10, -1, 0, 1, 10])
def test_scroll(controller, amount):
    controller.scroll(amount)
    assert fake_pyautogui.calls == [("scroll", amount)]


@pytest.mark.parametrize("amount", [1.5, "1", None])
def test_scroll_requires_integer(controller, amount):
    with pytest.raises(TypeError):
        controller.scroll(amount)  # type: ignore[arg-type]
