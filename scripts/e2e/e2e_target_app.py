from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional path that receives the exact submitted message.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    root.title("GUI Agent E2E Target")
    root.geometry("760x470")

    # Bring the deterministic test target to the foreground when launched
    # from the Week 4 task suite. The topmost flag is temporary so the app
    # still behaves like an ordinary desktop window afterwards.
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))

    title = tk.Label(
        root,
        text="GUI Agent Message Target",
        font=("Segoe UI", 22),
    )
    title.pack(pady=(24, 10))

    hint = tk.Label(
        root,
        text="Use E2E_TARGET to focus the message box, then press SEND.",
        font=("Segoe UI", 12),
    )
    hint.pack(pady=(0, 10))

    status_var = tk.StringVar(value="STATUS WAITING")
    sent_var = tk.StringVar(value="")
    entry_var = tk.StringVar()

    entry = tk.Entry(
        root,
        textvariable=entry_var,
        font=("Microsoft YaHei", 17),
        width=42,
    )

    def on_target_click() -> None:
        entry.delete(0, tk.END)
        entry.focus_set()
        status_var.set("STATUS READY")
        sent_var.set("")

    target_button = tk.Button(
        root,
        text="E2E_TARGET",
        font=("Segoe UI", 18),
        command=on_target_click,
        width=20,
        height=2,
    )
    target_button.pack(pady=8)

    entry.pack(pady=12)

    def on_send() -> None:
        text = entry_var.get().strip()

        if not text:
            status_var.set("STATUS EMPTY")
            sent_var.set("")
            entry.focus_set()
            return

        if args.receipt is not None:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(text, encoding="utf-8")

        # Keep the OCR oracle deliberately simple and stable. The exact
        # submitted payload is shown separately and verified through the
        # receipt file by the deterministic E2E harness.
        status_var.set("MESSAGE SENT")
        sent_var.set(f"SENT TEXT: {text}")

    send_button = tk.Button(
        root,
        text="SEND",
        font=("Segoe UI", 18),
        command=on_send,
        width=14,
        height=2,
    )
    send_button.pack(pady=8)

    status = tk.Label(
        root,
        textvariable=status_var,
        font=("Microsoft YaHei", 20, "bold"),
    )
    status.pack(pady=(14, 6))

    sent = tk.Label(
        root,
        textvariable=sent_var,
        font=("Microsoft YaHei", 15),
    )
    sent.pack(pady=(0, 12))

    root.after(200, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
