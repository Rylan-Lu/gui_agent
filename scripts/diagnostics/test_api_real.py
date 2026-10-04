from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from gui_agent.models.api_model import APIModelClient
from gui_agent.models.base import ModelRequest

import argparse
import getpass
import json
import os
import tempfile
import time

from pathlib import Path
from urllib.error import HTTPError, URLError


def create_test_image(path: Path) -> None:
    """Create a synthetic image without desktop information."""

    image = Image.new("RGB", (640, 180), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default(size=60)

    draw.text(
        (60, 45),
        "TEST-42",
        fill="black",
        font=font,
    )

    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--endpoint",
        required=True,
        help="Official HTTPS Chat Completions endpoint",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Multimodal model ID",
    )

    args = parser.parse_args()

    if not os.environ.get("GUI_AGENT_API_KEY"):
        os.environ["GUI_AGENT_API_KEY"] = getpass.getpass(
            "Enter API Key (hidden): "
        )

    client = APIModelClient(
        endpoint=args.endpoint,
        model_id=args.model,
        timeout=60.0,
    )

    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / "test.png"

        create_test_image(image_path)

        request = ModelRequest(
            prompt=(
                "Read the text in this image. "
                "Reply with only the text you see."
            ),
            image_path=image_path,
        )

        print("===== Real Multimodal API Test =====")
        print("Model:", args.model)
        print("Sending one API request...")

        start = time.perf_counter()

        try:
            response = client.generate(request)

        except HTTPError as exc:
            elapsed = time.perf_counter() - start

            raw_body = exc.read()

            body = raw_body.decode(
                "utf-8",
                errors="replace",
            ).strip()

            print()
            print("===== HTTP ERROR =====")
            print("Status:", exc.code)
            print("Reason:", exc.reason)
            print(
                "Elapsed:",
                round(elapsed, 2),
                "seconds",
            )

            if body:
                try:
                    parsed = json.loads(body)

                    body = json.dumps(
                        parsed,
                        ensure_ascii=False,
                        indent=2,
                    )
                except json.JSONDecodeError:
                    pass

                print("Response body:")
                print(body[:4000])

            raise SystemExit(1)

        except URLError as exc:
            elapsed = time.perf_counter() - start

            print()
            print("===== NETWORK ERROR =====")
            print("Reason:", exc.reason)
            print(
                "Elapsed:",
                round(elapsed, 2),
                "seconds",
            )

            raise SystemExit(1)

        elapsed = time.perf_counter() - start

    print("Response model:", response.model_name)
    print("Response:", repr(response.text))
    print("Elapsed:", round(elapsed, 2), "seconds")

    if (
        isinstance(response.text, str)
        and "TEST-42" in response.text.upper()
    ):
        print("\nMULTIMODAL API TEST: PASS")
    else:
        print("\nMULTIMODAL API TEST: NEEDS REVIEW")


if __name__ == "__main__":
    main()