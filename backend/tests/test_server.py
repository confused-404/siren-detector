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
        del scope, receive
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"not found"})


class _FakeDetector:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.started = False
        self.stopped = False
        self.failure: RuntimeError | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_status(self) -> dict[str, int | str]:
        if self.failure is not None:
            raise self.failure
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
    server: Any = importlib.import_module("server")

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
    server: Any = importlib.import_module("server")

    async def run_lifespan() -> None:
        async with server.lifespan(server.app):
            with pytest.raises(HTTPException) as exc_info:
                server.status()
            assert exc_info.value.status_code == 503
            assert "Detector unavailable" in exc_info.value.detail
            assert "missing model" in exc_info.value.detail

    asyncio.run(run_lifespan())


def test_build_detector_config_uses_environment_override(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)
    monkeypatch.setenv("SIREN_MODEL_PATH", "/tmp/custom-model.h5")

    sys.modules.pop("server", None)
    server: Any = importlib.import_module("server")

    assert server.build_detector_config().model_path == "/tmp/custom-model.h5"


def test_status_raises_503_when_detector_not_initialized(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server: Any = importlib.import_module("server")
    server.detector = None
    server.startup_error = None

    with pytest.raises(HTTPException) as exc_info:
        server.status()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Detector unavailable: not initialized"


def test_status_raises_503_when_detector_becomes_unhealthy(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server: Any = importlib.import_module("server")
    server.startup_error = None
    server.detector = _FakeDetector(_FakeDetectorConfig())
    server.detector.failure = RuntimeError("microphone failed")

    with pytest.raises(HTTPException) as exc_info:
        server.status()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "microphone failed"


def test_api_status_route_is_registered(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server: Any = importlib.import_module("server")

    status_routes = [
        route
        for route in server.app.routes
        if getattr(route, "path", None) == "/api/status"
        and "GET" in getattr(route, "methods", set())
    ]

    assert len(status_routes) == 1
    assert status_routes[0].endpoint is server.status


def test_frontend_static_mount_is_registered(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    fake_live_detector: Any = types.ModuleType("live_detector")
    fake_live_detector.LiveDetector = _FakeDetector
    fake_live_detector.DetectorConfig = _FakeDetectorConfig
    monkeypatch.setitem(sys.modules, "live_detector", fake_live_detector)

    sys.modules.pop("server", None)
    server: Any = importlib.import_module("server")

    mount_routes = [
        route for route in server.app.routes if getattr(route, "name", None) == "frontend"
    ]

    assert len(mount_routes) == 1
