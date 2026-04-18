from __future__ import annotations

import importlib
import sys
from pathlib import Path

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

def test_server_status_and_lifecycle_use_detector(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setattr("fastapi.staticfiles.StaticFiles", _FakeStaticFiles)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    import live_detector

    monkeypatch.setattr(live_detector, "LiveDetector", _FakeDetector)

    sys.modules.pop("server", None)
    server = importlib.import_module("server")

    server.startup()
    response = server.status()
    server.shutdown()

    assert response == {"sound": "h", "direction": -1}
    assert server.detector.started is True
    assert server.detector.stopped is True
