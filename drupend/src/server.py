from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import random
from pathlib import Path
import os

app = FastAPI()

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "app" / "dist"

def get_current_status():
    sounds = ['s', 'h', 'n']
    dirs = [-1, 0, 1]
    return {
        "sound": random.choice(sounds),
        "direction": random.choice(dirs)
    }

@app.get("/api/status")
def status():
    return get_current_status()

if not DIST_DIR.exists():
    raise RuntimeError(f"Frontend dist not found at: {DIST_DIR}")

app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
