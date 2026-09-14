from __future__ import annotations

_TORCH_GPU_INITIALIZED = False


def initialize_torch_gpu_runtime() -> None:
    """Initialize Torch CUDA/cuDNN before Paddle is imported.

    The tested Windows GPU setup is import-order sensitive: Torch must load
    before Paddle/PaddleOCR to avoid DLL conflicts. The function is idempotent.
    """
    global _TORCH_GPU_INITIALIZED

    if _TORCH_GPU_INITIALIZED:
        return

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available.")

    device = torch.device("cuda:0")
    x = torch.zeros((1, 1, 8, 8), device=device)
    weight = torch.zeros((1, 1, 3, 3), device=device)

    F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()

    _TORCH_GPU_INITIALIZED = True
