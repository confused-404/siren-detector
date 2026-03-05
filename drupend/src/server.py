from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import random
from pathlib import Path
import os
from live_detector import LiveDetector, DetectorConfig

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "app" / "dist"

cfg = DetectorConfig()
detector = LiveDetector(cfg)

@app.on_event("startup")
def startup():
    detector.start()

@app.on_event("shutdown")
def shutdown():
    detector.stop()

@app.get("/api/status")
def status():
    return detector.get_status()

if not DIST_DIR.exists():
    raise RuntimeError(f"Frontend dist not found at: {DIST_DIR}")

app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
