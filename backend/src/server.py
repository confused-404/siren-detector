from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from live_detector import LiveDetector, DetectorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "app" / "dist"

cfg = DetectorConfig(
    model_path=str(REPO_ROOT / "backend" / "src" / "siren_detector" / "ai" / "trained_car_alert_model.h5")
)
detector = LiveDetector(cfg)

@asynccontextmanager
async def lifespan(_: FastAPI):
    print("SERVER STARTUP: starting detector...")
    detector.start()
    print("SERVER STARTUP: detector.start() returned")
    try:
        yield
    finally:
        detector.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/api/status")
def status() -> Dict[str, Union[str, int]]:
    return detector.get_status()

if not DIST_DIR.exists():
    raise RuntimeError(f"Frontend dist not found at: {DIST_DIR}")

app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
