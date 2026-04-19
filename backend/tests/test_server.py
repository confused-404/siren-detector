from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException


class _FakeStaticFiles:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def __call__(self, scope, receive, send) -> None:
        del scope, receive, send


class _FakeDetector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_status(self) -> dict[str, int | str]:
        return {"sound": "h", "direction": -1}


class _FakeDetectorConfig:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def test_server_status_and_lifecycle_use_detector(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server = importlib.import_module("server")

    async def run_lifespan() -> tuple[dict[str, int | str], _FakeDetector]:
        async with server.lifespan(server.app):
            detector = server.detector
            assert detector is not None
            return server.status(), detector

    status, detector = asyncio.run(run_lifespan())
    assert status == {"sound": "h", "direction": -1}
    assert detector.started is True
    assert detector.stopped is True


def test_server_status_returns_503_when_detector_startup_fails(monkeypatch) -> None:
    class _FailingDetector:
        def __init__(self, cfg) -> None:
            del cfg
            raise FileNotFoundError("missing model")

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FailingDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server = importlib.import_module("server")

    async def run_lifespan() -> None:
        async with server.lifespan(server.app):
            with pytest.raises(HTTPException) as exc_info:
                server.status()
            assert exc_info.value.status_code == 503
            assert "Detector unavailable" in exc_info.value.detail
            assert "missing model" in exc_info.value.detail

    asyncio.run(run_lifespan())
