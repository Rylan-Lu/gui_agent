import ctypes
import time
from ctypes import wintypes


# ============================================================
# Windows constants
# ============================================================

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


# ULONG_PTR:
# 32-bit Windows -> 32 bit
# 64-bit Windows -> 64 bit
#
# WPARAM has the pointer-sized integer layout required here.
ULONG_PTR = wintypes.WPARAM


# ============================================================
# Windows INPUT structures
# ============================================================

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)

    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUTUNION),
    ]


# ============================================================
# Windows API
# ============================================================

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True,
)

SendInput = user32.SendInput

SendInput.argtypes = (
    wintypes.UINT,
    ctypes.POINTER(INPUT),
    ctypes.c_int,
)

SendInput.restype = wintypes.UINT


# ============================================================
# Unicode input
# ============================================================

def _send_utf16_unit(
    code_unit: int,
) -> None:
    """
    Send one UTF-16 code unit using Windows SendInput.

    KEYEVENTF_UNICODE requires wVk=0 and places the UTF-16
    code unit in wScan.

    One key-down event and one key-up event are emitted.
    """
    key_down = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=code_unit,
            dwFlags=KEYEVENTF_UNICODE,
            time=0,
            dwExtraInfo=0,
        ),
    )

    key_up = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(
            wVk=0,
            wScan=code_unit,
            dwFlags=(
                KEYEVENTF_UNICODE
                | KEYEVENTF_KEYUP
            ),
            time=0,
            dwExtraInfo=0,
        ),
    )

    inputs = (INPUT * 2)(
        key_down,
        key_up,
    )

    sent = SendInput(
        len(inputs),
        inputs,
        ctypes.sizeof(INPUT),
    )

    if sent != len(inputs):
        error_code = ctypes.get_last_error()

        raise RuntimeError(
            f"SendInput sent {sent}/{len(inputs)} "
            f"keyboard events. "
            f"Windows error code: {error_code}. "
            f"INPUT size: {ctypes.sizeof(INPUT)}. "
            "Input may have been blocked by Windows UIPI "
            "or integrity-level restrictions."
        )


def type_unicode_text(
    text: str,
    *,
    interval: float = 0.0,
) -> None:
    """
    Type Unicode text using Windows SendInput.

    Each Python character is encoded as UTF-16 LE. Characters
    outside the Basic Multilingual Plane, such as emoji, produce
    a UTF-16 surrogate pair and therefore two code units.

    The interval is applied once per Python character rather than
    once per UTF-16 code unit, so surrogate pairs are sent without
    an artificial delay between their two halves.
    """
    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    if interval < 0:
        raise ValueError(
            "interval must be non-negative"
        )

    for char in text:
        encoded = char.encode(
            "utf-16-le"
        )

        for index in range(
            0,
            len(encoded),
            2,
        ):
            code_unit = int.from_bytes(
                encoded[
                    index:index + 2
                ],
                byteorder="little",
            )

            _send_utf16_unit(
                code_unit
            )

        if interval > 0:
            time.sleep(interval)