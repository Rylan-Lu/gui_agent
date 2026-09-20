from pathlib import Path
from time import perf_counter

from gui_agent.models.base import ModelRequest
from gui_agent.models.local_vlm import LocalVLMClient


ROOT = Path(__file__).resolve().parents[1]

SCREENAGENT_DIR = (
    ROOT.parent
    / "ScreenAgent_reference"
    / "data"
    / "ScreenAgent"
    / "train"
)


def main():
    images = sorted(SCREENAGENT_DIR.rglob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            "No ScreenAgent screenshots found."
        )

    image_path = images[0]

    print("Image:", image_path)

    client = LocalVLMClient(
        max_new_tokens=128,
    )

    request = ModelRequest(
        prompt=(
            "Describe the visible GUI. "
            "Identify the application, main interface "
            "elements, and any clearly readable text. "
            "Do not invent details."
        ),
        image_path=image_path,
    )

    start = perf_counter()

    response = client.generate(request)

    elapsed = perf_counter() - start

    print("\n===== VLM RESULT =====")
    print("Model:", response.model_name)
    print("Response:", response.text)
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()