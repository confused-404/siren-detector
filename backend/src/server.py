import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from live_detector import DetectorConfig, LiveDetector
from status_types import StatusResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "app" / "dist"
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "backend" / "src" / "siren_detector" / "ai" / "trained_car_alert_model.tflite"
)

detector: LiveDetector | None = None
startup_error: str | None = None


def build_detector_config() -> DetectorConfig:
    model_path = os.environ.get("SIREN_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    return DetectorConfig(model_path=model_path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global detector, startup_error

    logger.info("Starting detector during server startup")
    detector = None
    startup_error = None
    try:
        detector = LiveDetector(build_detector_config())
        detector.start()
        logger.info("Detector startup completed")
    except Exception as exc:
        startup_error = f"Detector unavailable: {exc}"
        logger.exception("Detector failed to start")
    try:
        yield
    finally:
        if detector is not None:
            detector.stop()
        detector = None


app = FastAPI(lifespan=lifespan)


@app.get("/api/status")
def status() -> StatusResponse:
    if startup_error is not None:
        raise HTTPException(status_code=503, detail=startup_error)
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector unavailable: not initialized")
    try:
        return detector.get_status()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if not DIST_DIR.exists():
    raise RuntimeError(f"Frontend dist not found at: {DIST_DIR}")

app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
