from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import random

app = FastAPI()

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

app.mount("/", StaticFiles(directory="../app/dist", html=True), name="frontend")
