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

        # Keep PyAutoGUI's emergency failsafe enabled.
        pyautogui.FAILSAFE = True

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._width, self._height

    def move_to(
        self,
        point: Point,
        *,
        duration: float = 0.3,
    ) -> None:
        self._validate_duration(duration)

        x, y = self._prepare_point(point)

        pyautogui.moveTo(
            x,
            y,
            duration=duration,
        )

    def click(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.2,
    ) -> None:
        self.move_to(
            point,
            duration=duration,
        )

        pyautogui.click(
            button=button,
        )

    def double_click(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.2,
        interval: float = 0.15,
    ) -> None:
        # Validate interval before moving the mouse so invalid input
        # does not cause a partial side effect.
        self._validate_interval(interval)

        self.move_to(
            point,
            duration=duration,
        )

        pyautogui.doubleClick(
            button=button,
            interval=interval,
        )

    def drag_to(
        self,
        point: Point,
        *,
        button: MouseButton = "left",
        duration: float = 0.5,
    ) -> None:
        self._validate_duration(duration)

        x, y = self._prepare_point(point)

        pyautogui.dragTo(
            x,
            y,
            duration=duration,
            button=button,
        )

    def type_text(
        self,
        text: str,
        *,
        interval: float = 0.02,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self._validate_interval(interval)

        # Use the Windows SendInput Unicode path for both ASCII
        # and non-ASCII text so behavior does not depend on IME state.
        type_unicode_text(
            text,
            interval=interval,
        )

    def press(
        self,
        key: str,
        *,
        presses: int = 1,
        interval: float = 0.0,
    ) -> None:
        if presses <= 0:
            raise ValueError(
                "presses must be a positive integer"
            )

        self._validate_interval(interval)

        pyautogui.press(
            key,
            presses=presses,
            interval=interval,
        )

    def hotkey(
        self,
        *keys: str,
        interval: float = 0.5,
    ) -> None:
        if not keys:
            raise ValueError(
                "at least one key must be provided"
            )

        self._validate_interval(interval)

        pyautogui.hotkey(
            *keys,
            interval=interval,
        )

    def scroll(
        self,
        amount: int,
    ) -> None:
        if not isinstance(amount, int):
            raise TypeError(
                "amount must be an integer"
            )

        pyautogui.scroll(amount)

    def _prepare_point(
        self,
        point: Point,
    ) -> tuple[int, int]:
        """
        Validate and convert a floating-point desktop point into
        integer pixel coordinates.

        The point is checked both before and after rounding.

        The second check is required because a valid floating-point
        coordinate near the right or bottom edge may round outside
        the valid screen range.

        Example on a 2560-pixel-wide screen:

            x = 2559.6
            round(x) = 2560

        Although 2559.6 is smaller than 2560, the resulting integer
        coordinate 2560 is outside the valid range 0..2559.
        """
        self._validate_point(point)

        x, y = self._round_point(point)

        if not (
            0 <= x < self._width
            and 0 <= y < self._height
        ):
            raise ControlError(
                f"Rounded point {(x, y)} is out of screen bounds "
                f"for original point {point}"
            )

        return x, y

    def _validate_point(
        self,
        point: Point,
    ) -> None:
        x, y = point

        if not (
            0 <= x < self._width
            and 0 <= y < self._height
        ):
            raise ControlError(
                f"Point {point} is out of screen bounds"
            )

    @staticmethod
    def _validate_duration(
        duration: float,
    ) -> None:
        if duration < 0:
            raise ValueError(
                "duration must be a non-negative number"
            )

    @staticmethod
    def _validate_interval(
        interval: float,
    ) -> None:
        if interval < 0:
            raise ValueError(
                "interval must be a non-negative number"
            )

    @staticmethod
    def _round_point(
        point: Point,
    ) -> tuple[int, int]:
        return (
            round(point[0]),
            round(point[1]),
        )