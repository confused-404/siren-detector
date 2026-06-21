#!/bin/bash
set -e

export HOME=/home/andy
export PATH="/home/andy/.local/bin:/home/andy.nvm/versions/node/v20.20.0/bin:/usr/local/bin:/usr/bin:/bin"

echo "Starting Siren Detector Demo..."

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

APP_DIR="$REPO_DIR/app"
BACKEND_DIR="$REPO_DIR/drupend"

# echo "Building frontend..."
# cd "$APP_DIR"
# /home/andy/.nvm/versions/node/v20.20.0/bin/npm run build

echo "Hoping frontend is already built..."

echo "Starting backend server..."
cd "$BACKEND_DIR"

/home/andy/.cache/pypoetry/virtualenvs/siren-detector-y99sc-Hf-py3.11/bin/python -m uvicorn server:app \
  --app-dir src \
  --host 0.0.0.0 \
  --port 3000
