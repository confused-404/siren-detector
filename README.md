# Siren Detector

A real-time audio detection system that alerts deaf drivers to sirens and honks using machine learning and directional audio localization.

## Overview

Siren Detector is an embedded system designed to detect and locate emergency sirens and car honks in real-time. Running on a Raspberry Pi 5 with dual microphones, it uses a convolutional neural network to classify audio events and a GCC-PHAT algorithm to determine sound direction (left, center, or right). The system provides visual alerts through a lightweight web-based dashboard.

### Key Features

- **Real-time Audio Detection**: Continuously processes 1-second audio clips from dual I2S INMP441 microphones
- **Multi-class Classification**: Detects three sound categories: sirens, honks, and ambient noise
- **Directional Localization**: Uses GCC-PHAT time-difference-of-arrival (TDOA) analysis to identify sound direction
- **Lightweight Architecture**: Optimized for edge deployment on Raspberry Pi
- **Web Dashboard**: Simple, responsive visual interface for real-time alerts
- **Automatic Startup**: Boots automatically on Raspberry Pi with built-in hotspot

## Technical Architecture

### System Components

```
┌─────────────────────────────────────┐
│   Dual I2S Microphones              │
│   (INMP441 on Raspberry Pi 5)       │
└──────────────┬──────────────────────┘
               │ arecord (1-sec clips)
               ▼
┌─────────────────────────────────────┐
│   Audio Processing Backend          │
│   (Python + FastAPI)                │
│  ┌──────────────────────────────┐   │
│  │ Log-Spectrogram Extraction   │   │
│  │ (STFT + Log-magnitude)       │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ Log-Spec CNN Model           │   │
│  │ (TensorFlow/TFLite)          │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ GCC-PHAT Direction Estimation│   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │ JSON API
               ▼
┌─────────────────────────────────────┐
│   Web Frontend Dashboard            │
│   (Vanilla JavaScript + Vite)       │
│   Visual Direction Indicator        │
└─────────────────────────────────────┘
```

### Machine Learning Model

Initially, a multilayer perceptron (MLP) was trained on raw audio frequencies, but achieved only ~85% accuracy. The approach was switched to a **log-spectrogram CNN** which significantly improved performance by:

1. Converting raw waveform to STFT representation
2. Computing log-magnitude spectrograms for better frequency resolution
3. Using convolutional layers to capture temporal and spectral patterns
4. Training on balanced dataset of siren, honk, and noise samples

The model is exported as TensorFlow/TFLite for efficient inference on edge devices.

### Audio Processing Pipeline

1. **Recording**: `arecord` continuously captures 1-second audio clips at 16 kHz, 2 channels from device `plughw:2,0`
2. **Feature Extraction**: 
   - STFT with 512-sample frame length, 128-sample hop length
   - Log-magnitude spectrogram computation
   - Peak normalization (0.5 limit) to handle varying input levels
3. **Inference**: Pre-trained CNN classifies into 3 classes
4. **Direction Estimation**: GCC-PHAT algorithm on dual channels computes time delay of arrival, converted to directional indicator (-1: left, 0: center, +1: right)

## Hardware Requirements

- **Raspberry Pi 5** (or compatible SBC with audio support)
- **2× I2S INMP441 Microphones** (or similar I2S-compatible digital microphones)
- Power supply and network connectivity
- Optional: enclosure for weatherproofing

## Software Stack

### Backend
- **Python 3.11+**
- **FastAPI** - Web framework for API
- **TensorFlow/TFLite** - Deep learning inference
- **NumPy/Pandas** - Numerical computing

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **Vite** - Build tool
- **CORS** - Cross-origin resource sharing

## Getting Started

### Prerequisites

- Python 3.11 (required for TFLite runtime compatibility)
- Node.js 16+ (for frontend build)
- Poetry (Python dependency manager)
- Audio device support on target hardware

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd siren-detector
   ```

2. **Backend setup**
   ```bash
   cd backend
   poetry install
   ```

3. **Frontend setup**
   ```bash
   cd ../app
   npm install
   ```

### Running the Demo

From the project root:

```bash
./run_demo.sh
```

This script will:
1. Build the frontend with Vite
2. Start the FastAPI backend server on `http://0.0.0.0:3000`
3. Serve the dashboard at the root path

The backend will:
- Load the pre-trained CNN model
- Initialize audio recording on startup
- Expose a `/api/status` endpoint returning detection results
- Stream detection data to the dashboard

### Manual Backend Startup

```bash
cd backend
poetry run uvicorn server:app --app-dir src --host 0.0.0.0 --port 3000
```

If trained model lives outside the repository, set `SIREN_MODEL_PATH` before starting the backend:

```bash
cd backend
SIREN_MODEL_PATH=/absolute/path/to/trained_car_alert_model.h5 \
poetry run uvicorn server:app --app-dir src --host 0.0.0.0 --port 3000
```

When the model is missing or invalid, the server stays up and `GET /api/status` returns `503` with a detector error.

### Backend Tests

The backend includes a small `pytest` unit test suite for API lifecycle behavior, detector helpers, and spectrogram utilities.

```bash
cd backend
poetry run pytest
```

Tests live in `backend/tests/`.

### Code Quality Checks

The project includes baseline linting, formatting, and type-checking for both the backend and frontend.

Backend:

```bash
cd backend
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy
```

Frontend:

```bash
cd app
npm run lint
npm run format
npm run typecheck
```

### Manual Frontend Development

```bash
cd app
npm run dev      # Development server with hot reload
npm run build    # Production build to dist/
npm run preview  # Preview production build
```

## API Endpoints

### `GET /api/status`

Returns the current detection status.

**Response:**
```json
{
  "sound": "s|h|n",
  "direction": -1|0|1
}
```

Where:
- `sound`: Sound type ("s" = siren, "h" = honk, "n" = noise)
- `direction`: Direction indicator (-1 = left, 0 = center, +1 = right)

## Configuration

Audio and model parameters can be configured in [backend/src/live_detector.py](backend/src/live_detector.py):

```python
DetectorConfig(
    model_path="...",           # Path to trained model
    sample_rate=16000,          # Audio sample rate (Hz)
    channels=2,                 # Number of input channels
    block_seconds=1.0,          # Processing block size
    hop_seconds=0.25,           # Overlap for sliding window
    peak_limit=0.5,             # Normalization threshold
    frame_length=512,           # STFT frame size
    frame_step=128,             # STFT hop size
    fft_length=512,             # FFT size
    mic_distance_m=0.1,         # Distance between microphones
    speed_of_sound=343.0,       # Speed of sound (m/s)
    direction_deadband_deg=10.0,# Direction threshold
    arecord_device="plughw:2,0",# ALSA device name
)
```

## Training

To retrain the model with custom data:

1. **Prepare dataset** using [backend/scripts/record_dataset.py](backend/scripts/record_dataset.py)

2. **Train model** using [backend/scripts/trainer.py](backend/scripts/trainer.py)
   ```bash
   cd backend/scripts
   poetry run python trainer.py
   ```

The trainer will:
- Load log-spectrograms from your dataset
- Train the CNN model with early stopping
- Evaluate confusion matrix and classification metrics
- Save the model as `trained_car_alert_model.h5`

## Dashboard

The web dashboard provides a real-time visual interface:

- **Left Panel** - Highlights when sound is detected from the left
- **Center Panel** - Highlights when sound is directly ahead
- **Right Panel** - Highlights when sound is detected from the right
- **Color Coding**:
  - Blue = Siren detected
  - Yellow = Honk detected
  - White = Ambient noise

The dashboard updates via polling `/api/status` endpoint and is optimized for low bandwidth and minimal resource usage.

## File Structure

```
siren-detector/
├── run_demo.sh                      # Quick start script
├── app/                             # Frontend (Vanilla JS + Vite)
│   ├── index.html
│   ├── main.js
│   ├── style.css
│   └── package.json
└── backend/                         # Python backend
    ├── pyproject.toml
    ├── scripts/                     # Training and model creation scripts
    │   ├── record_dataset.py       # Dataset recording utility
    │   ├── trainer.py              # Main training script
    │   ├── utils.py                # Training utility functions
    │   └── helpers/                # Training helper modules
    │       ├── create_model.py     # Model architecture definitions
    │       ├── middleman.py        # Data loading and preprocessing
    │       └── training.py         # Training utilities and functions
    ├── src/
    │   ├── server.py               # FastAPI server
    │   ├── live_detector.py        # Real-time detection engine
    │   ├── utils.py                # Runtime utility functions
    │   └── siren_detector/
    │       └── trained_car_alert_model.h5  # Pre-trained model
    └── tests/
```

## Raspberry Pi Setup (Auto-Boot)

### Automatic Startup

To enable automatic startup on Raspberry Pi:

1. Add `run_demo.sh` to cron or systemd service
2. The system automatically establishes a hotspot for connectivity
3. Access dashboard at device IP/hotspot on port 3000

## Troubleshooting

### Audio Device Not Found

Check ALSA device configuration:
```bash
arecord -l
```

Update `arecord_device` in `DetectorConfig` to match your hardware.

### Model Not Loading

Ensure the model file exists at the path specified in `DetectorConfig.model_path`. Verify model is a valid `.h5` file compatible with your TensorFlow version.

### High CPU Usage

The processing window can be tuned via:
- Increase `block_seconds` for larger processing chunks
- Reduce frontend polling frequency
- Use TFLite model for lower precision/faster inference

## Acknowledgments

- Audio recording using [arecord](https://linux.die.net/man/1/arecord)
- Direction estimation based on [GCC-PHAT](https://dsp.stackexchange.com/questions/74574/understanding-gcc-phat-as-a-feature) algorithm

**Note**: This system is designed to provide audio information that is not a substitute for visual attention while driving. Always obey traffic laws and remain alert to all road conditions.
