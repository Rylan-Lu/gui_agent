from __future__ import annotations

import sys
from types import SimpleNamespace

from gui_agent.models.local_vlm import LocalVLMClient


class _FakeCuda:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def synchronize(self) -> None:
        self.sync_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def test_unload_releases_references_and_cuda_cache(monkeypatch) -> None:
    client = LocalVLMClient()
    client.model = object()
    client.processor = object()

    cuda = _FakeCuda()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda),
    )

    client.unload()

    assert client.model is None
    assert client.processor is None
    assert cuda.sync_calls == 1
    assert cuda.empty_cache_calls == 1


def test_unload_is_noop_when_not_loaded(monkeypatch) -> None:
    client = LocalVLMClient()
    cuda = _FakeCuda()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda),
    )

    client.unload()

    assert cuda.sync_calls == 0
    assert cuda.empty_cache_calls == 0
