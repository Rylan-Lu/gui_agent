from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    """
    Unified input for a language or vision-language model.
    """

    prompt: str

    # Optional screenshot for multimodal models.
    image_path: Path | None = None

    # Optional system instruction.
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must not be empty")

        if self.image_path is not None and not isinstance(
            self.image_path, Path
        ):
            raise TypeError("image_path must be a Path or None")


@dataclass(frozen=True)
class ModelResponse:
    """
    Unified output returned by a model.
    """

    text: str
    model_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("response text must not be empty")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must not be empty")


class ModelClient(Protocol):
    """
    Common interface for local and remote models.
    """

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate a response from the supplied request.
        """
        ...