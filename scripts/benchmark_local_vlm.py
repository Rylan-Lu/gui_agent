from pathlib import Path
from time import perf_counter

import torch

from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import LocalVLMClient


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT.parent
    / "ScreenAgent_reference"
    / "data"
    / "ScreenAgent"
    / "train"
)


def main():
    images = sorted(DATA_ROOT.rglob("*.jpg"))

    if not images:
        raise FileNotFoundError("No screenshots found.")

    image_path = images[0]

    client = LocalVLMClient(max_new_tokens=128)

    request = ModelRequest(
        prompt="Briefly describe the visible GUI.",
        image_path=image_path,
    )

    # Load the model only once.
    start = perf_counter()
    client.load()
    load_time = perf_counter() - start

    print(f"Model load: {load_time:.2f}s")

    # Run two inference calls with the same model instance.
    for i in range(2):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = perf_counter()

        response = client.generate(request)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = perf_counter() - start

        print(f"\nInference {i + 1}: {elapsed:.2f}s")
        print("Response:", response.text)

    if torch.cuda.is_available():
        print(
            "\nPeak allocated GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**3,
                2,
            ),
            "GiB",
        )


if __name__ == "__main__":
    main()