import tkinter as tk


def main() -> None:
    root = tk.Tk()
    root.title("GUI Agent E2E Target")
    root.geometry("700x350")

    title = tk.Label(
        root,
        text="GUI Agent E2E Test",
        font=("Segoe UI", 20),
    )
    title.pack(pady=30)

    status_var = tk.StringVar(
        value="STATUS: WAITING"
    )

    entry_var = tk.StringVar()

    entry = tk.Entry(
        root,
        textvariable=entry_var,
        font=("Microsoft YaHei", 16),
        width=40,
    )

    def on_target_click() -> None:
        entry.delete(0, tk.END)
        entry.focus_set()
        status_var.set("STATUS: READY")

    button = tk.Button(
        root,
        text="E2E_TARGET",
        font=("Segoe UI", 18),
        command=on_target_click,
        width=20,
        height=2,
    )
    button.pack(pady=10)

    entry.pack(pady=20)

    status = tk.Label(
        root,
        textvariable=status_var,
        font=("Microsoft YaHei", 14),
    )
    status.pack(pady=15)

    def update_status(*_) -> None:
        text = entry_var.get()

        if text:
            status_var.set(
                f"STATUS: {text}"
            )

    entry_var.trace_add(
        "write",
        update_status,
    )

    root.mainloop()


if __name__ == "__main__":
    main()