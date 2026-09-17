from pathlib import Path

from gui_agent.datasets.webarena import (
    load_webarena_file,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT.parent
    / "WebArena_reference"
    / "2.json"
)


def main():

    task = load_webarena_file(SOURCE)

    print("===== WebArena Real Data =====")

    print("Task ID:", task.task_id)
    print("Intent:", task.intent)
    print("Sites:", task.sites)
    print("Start URL:", task.start_url)

    print("Require login:", task.require_login)
    print("Require reset:", task.require_reset)

    print(
        "Evaluation fields:",
        list(task.evaluation.keys()),
    )

    print(
        "Reference action count:",
        task.reference_action_count,
    )

    assert task.task_id == 2
    assert task.reference_action_count == 2

    print("\nWEBARENA ADAPTER: PASS")


if __name__ == "__main__":
    main()