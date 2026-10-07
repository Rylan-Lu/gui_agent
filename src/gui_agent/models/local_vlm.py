from __future__ import annotations

import gc
import sys
from pathlib import Path

from gui_agent.models.base import (
    ModelRequest,
    ModelResponse,
)
from gui_agent.runtime.gpu_runtime import initialize_torch_gpu_runtime


class LocalVLMClient:
    """Local Qwen3-VL model adapter."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        max_new_tokens: int = 128,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor once."""

        if self.model is not None:
            return

        # Use the shared, validated Torch-first initialization path before
        # loading other GPU-heavy model stacks in this process.
        initialize_torch_gpu_runtime()

        import torch
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=True,
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=True,
        )

        model.eval()

        self.processor = processor
        self.model = model

    def unload(self) -> None:
        """
        Release local planner resources after one-shot planning.

        The unified desktop path uses the GPU again for PaddleOCR. Keeping
        several GiB of Qwen weights resident after planning can leave too
        little room for Paddle/cuBLAS workspaces. The client remains reusable:
        a later generate() call will load the model again.
        """

        if self.model is None and self.processor is None:
            return

        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None) if torch is not None else None

        if cuda is not None and cuda.is_available():
            cuda.synchronize()

        self.model = None
        self.processor = None
        gc.collect()

        if cuda is not None and cuda.is_available():
            cuda.empty_cache()

    @staticmethod
    def _build_messages(request: ModelRequest) -> list[dict]:
        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt,
            })

        content = []

        if request.image_path is not None:
            path = Path(request.image_path).resolve()

            if not path.is_file():
                raise FileNotFoundError(path)

            from PIL import Image

            with Image.open(path) as img:
                image = img.convert("RGB")

            content.append({
                "type": "image",
                "image": image,
            })

        content.append({
            "type": "text",
            "text": request.prompt,
        })

        messages.append({
            "role": "user",
            "content": content,
        })

        return messages

    def generate(self, request: ModelRequest) -> ModelResponse:
        import torch

        # Validate input before loading several GB of weights.
        messages = self._build_messages(request)

        self.load()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][prompt_length:]

        text = self.processor.batch_decode(
            [generated_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return ModelResponse(
            text=text,
            model_name=self.model_id,
        )
