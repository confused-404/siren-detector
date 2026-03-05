#!/bin/bash
set -e

echo "Starting Siren Detector Demo..."

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

APP_DIR="$REPO_DIR/app"
BACKEND_DIR="$REPO_DIR/drupend"

echo "Building frontend..."
cd "$APP_DIR"
npm run build

echo "Starting backend server..."
cd "$BACKEND_DIR"

poetry run uvicorn server:app \
  --app-dir src \
  --host 0.0.0.0 \
  --port 3000
