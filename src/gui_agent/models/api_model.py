from __future__ import annotations

import base64
import io
import json
import os

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

from gui_agent.models.base import ModelRequest, ModelResponse


class APIModelClient:
    """OpenAI-compatible Chat Completions adapter."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        timeout: float = 60.0,
    ):
        parsed = urlparse(endpoint)

        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/chat/completions")
        ):
            raise ValueError(
                "endpoint must be an HTTPS chat/completions URL"
            )

        if not model_id.strip():
            raise ValueError("model_id must not be empty")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.endpoint = endpoint
        self.model_id = model_id
        self.timeout = timeout

    @staticmethod
    def _encode_image(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)

        # Convert to PNG to avoid relying on filename extensions.
        with Image.open(path) as img:
            image = img.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        return f"data:image/png;base64,{encoded}"

    def _build_messages(self, request: ModelRequest) -> list[dict]:
        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt,
            })

        content = [{
            "type": "text",
            "text": request.prompt,
        }]

        if request.image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image(
                        request.image_path
                    )
                },
            })

        messages.append({
            "role": "user",
            "content": content,
        })

        return messages

    def generate(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get("GUI_AGENT_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GUI_AGENT_API_KEY is not configured"
            )

        payload = {
            "model": self.model_id,
            "messages": self._build_messages(request),
        }

        body = json.dumps(payload).encode("utf-8")

        http_request = Request(
            url=self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(
            http_request,
            timeout=self.timeout,
        ) as response:
            data = json.load(response)

        try:
            choice = data["choices"][0]

            if choice.get("finish_reason") == "length":
                raise RuntimeError("API output was truncated")

            text = choice["message"]["content"]
            model_name = data.get("model") or self.model_id

            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    "Invalid API response content"
                )

        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Invalid API response structure"
            ) from exc

        return ModelResponse(
            text=text,
            model_name=model_name,
        )