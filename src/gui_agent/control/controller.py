from typing import Literal, TypeAlias

import pyautogui

from gui_agent.control.windows_text import type_unicode_text

Point: TypeAlias = tuple[float, float]
MouseButton: TypeAlias = Literal["left", "right", "middle"]


class ControlError(Exception):
    pass


class Controller:
    def __init__(self) -> None:
        self._width, self._height = pyautogui.size()
        pyautogui.FAILSAFE = True

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._width, self._height

    def move_to(self, point: Point, *, duration: float = 0.3) -> None:
        self._validate_point(point)
        self._validate_duration(duration)
        x, y = self._round_point(point)
        pyautogui.moveTo(x, y, duration=duration)

    def click(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.2,
    ) -> None:
        self.move_to(point, duration=duration)
        pyautogui.click(button=button)

    def double_click(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.2,
        interval: float = 0.15,
    ) -> None:
        self.move_to(point, duration=duration)
        pyautogui.doubleClick(button=button, interval=interval)

    def drag_to(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.5,
    ) -> None:
        self._validate_point(point)
        self._validate_duration(duration)
        x, y = self._round_point(point)
        pyautogui.dragTo(x, y, duration=duration, button=button)

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if interval < 0:
            raise ValueError("interval must be a non-negative number")

        if text.isascii():
            pyautogui.write(text, interval=interval)
            return

        type_unicode_text(text)

    def press(
        self,
        key: str,
        *,
        presses: int = 1,
        interval: float = 0.0,
    ) -> None:
        if presses <= 0:
            raise ValueError("presses must be a positive integer")
        if interval < 0:
            raise ValueError("interval must be a non-negative number")

        pyautogui.press(key, presses=presses, interval=interval)

    def hotkey(self, *keys: str, interval: float = 0.5) -> None:
        if not keys:
            raise ValueError("at least one key must be provided")
        if interval < 0:
            raise ValueError("interval must be a non-negative number")

        pyautogui.hotkey(*keys, interval=interval)

    def scroll(self, amount: int) -> None:
        if not isinstance(amount, int):
            raise TypeError("amount must be an integer")
        pyautogui.scroll(amount)

    def _validate_point(self, point: Point) -> None:
        x, y = point
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise ControlError(f"Point {point} is out of screen bounds")

    @staticmethod
    def _validate_duration(duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must be a non-negative number")

    @staticmethod
    def _round_point(point: Point) -> tuple[int, int]:
        return round(point[0]), round(point[1])
