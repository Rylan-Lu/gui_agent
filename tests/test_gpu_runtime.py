import sys
import types

import pytest

import gui_agent.runtime.gpu_runtime as runtime


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.sync_calls = 0

    def is_available(self):
        return self.available

    def synchronize(self):
        self.sync_calls += 1


class FakeTorch(types.ModuleType):
    def __init__(self, available=True):
        super().__init__("torch")
        self.cuda = FakeCuda(available)
        self.zeros_calls = []

    def device(self, name):
        return name

    def zeros(self, shape, device=None):
        self.zeros_calls.append((shape, device))
        return {"shape": shape, "device": device}


@pytest.fixture(autouse=True)
def reset_runtime_flag():
    runtime._TORCH_GPU_INITIALIZED = False
    yield
    runtime._TORCH_GPU_INITIALIZED = False


def install_fake_torch(monkeypatch, available=True):
    fake_torch = FakeTorch(available)
    functional = types.ModuleType("torch.nn.functional")
    functional.conv_calls = []

    def conv2d(x, weight, padding=0):
        functional.conv_calls.append((x, weight, padding))
        return "ok"

    functional.conv2d = conv2d
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    fake_torch.nn = nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)
    return fake_torch, functional


def test_gpu_runtime_initializes_cuda_and_cudnn(monkeypatch):
    fake_torch, functional = install_fake_torch(monkeypatch)
    runtime.initialize_torch_gpu_runtime()
    assert len(fake_torch.zeros_calls) == 2
    assert len(functional.conv_calls) == 1
    assert fake_torch.cuda.sync_calls == 1
    assert runtime._TORCH_GPU_INITIALIZED is True


def test_gpu_runtime_is_idempotent(monkeypatch):
    fake_torch, functional = install_fake_torch(monkeypatch)
    runtime.initialize_torch_gpu_runtime()
    runtime.initialize_torch_gpu_runtime()
    assert len(fake_torch.zeros_calls) == 2
    assert len(functional.conv_calls) == 1
    assert fake_torch.cuda.sync_calls == 1


def test_gpu_runtime_rejects_missing_cuda(monkeypatch):
    fake_torch, functional = install_fake_torch(monkeypatch, available=False)
    with pytest.raises(RuntimeError):
        runtime.initialize_torch_gpu_runtime()
    assert fake_torch.zeros_calls == []
    assert functional.conv_calls == []
    assert runtime._TORCH_GPU_INITIALIZED is False
