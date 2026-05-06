import argparse
import csv
import datetime as dt
import logging
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ARECORD_DEVICE = "plughw:2,0"
RATE = 16000
CHANNELS = 2
FORMAT = "S32_LE"
DTYPE = np.int32
DURATION_S = 1

COMMANDS = {
    "sl": ("siren", "left"),
    "sc": ("siren", "center"),
    "sr": ("siren", "right"),
    "hl": ("honk", "left"),
    "hc": ("honk", "center"),
    "hr": ("honk", "right"),
    "n": ("noise", "none"),
}

HELP_TEXT = """Interactive labeling:
  sl = siren left     sc = siren center     sr = siren right
  hl = honk  left     hc = honk  center     hr = honk  right
  n  = noise/none
  q  = quit

How it works:
  - Run the script once
  - Start the sound first
  - Type a label (e.g., sl) and press Enter
  - It records exactly 1 second per label
"""


def wait_for_quit(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            s = input().strip().lower()
        except EOFError:
            stop_event.set()
            return
        if s == "q":
            stop_event.set()
            return


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def record_1s_raw_int32_stereo() -> np.ndarray:
    """Returns int32 array shape (RATE, 2)"""
    cmd = [
        "arecord",
        "-D",
        ARECORD_DEVICE,
        "-f",
        FORMAT,
        "-r",
        str(RATE),
        "-c",
        str(CHANNELS),
        "-d",
        str(DURATION_S),
        "-t",
        "raw",
        "-q",
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error("arecord failed. Output:\n%s", e.output.decode(errors="replace"))
        raise

    x = np.frombuffer(raw, dtype=DTYPE)
    expected = RATE * CHANNELS * DURATION_S
    if x.size != expected:
        raise RuntimeError(f"Expected {expected} samples, got {x.size}.")
    return x.reshape(-1, CHANNELS)  # (16000, 2)


def int32_to_float32_unit(x_int32: np.ndarray) -> np.ndarray:
    return x_int32.astype(np.float32) / 2147483648.0


def shared_rms_normalize(
    x: np.ndarray,
    target_rms: float = 0.08,
    max_gain: float = 20.0,
    eps: float = 1e-8,
) -> np.ndarray:
    rms = np.sqrt(np.mean(x[:, 0] ** 2 + x[:, 1] ** 2) / 2.0)
    gain = target_rms / max(rms, eps)
    gain = min(gain, max_gain)
    return np.clip(x * gain, -1.0, 1.0)


def ensure_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["file", "event", "direction", "ts", "device", "rate", "channels", "normalized"]
            )


def append_manifest(
    manifest_path: Path,
    file_path: Path,
    event: str,
    direction: str,
    normalized: bool,
) -> None:
    with manifest_path.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                str(file_path),
                event,
                direction,
                dt.datetime.now().isoformat(timespec="seconds"),
                ARECORD_DEVICE,
                RATE,
                CHANNELS,
                int(normalized),
            ]
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Record 1-second stereo clips labeled by event + direction."
    )
    p.add_argument("--out", default="dataset", help="Output directory")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Shared RMS normalize (same gain both channels)",
    )
    p.add_argument("--count", type=int, default=0, help="Stop after N clips (0 = until quit)")
    p.add_argument(
        "--label",
        default=None,
        help="Continuous mode label key (sl/sc/sr/hl/hc/hr/n). Type 'q' + Enter to stop.",
    )
    args = p.parse_args()

    out_root = Path(args.out)
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.csv"
    ensure_manifest(manifest)

    logger.info("\n%s", HELP_TEXT)

    saved = 0

    if args.label is not None:
        key = args.label.strip().lower()
        if key not in COMMANDS:
            logger.error("Unknown --label. Must be one of: %s", ", ".join(COMMANDS.keys()))
            sys.exit(2)

        event, direction = COMMANDS[key]
        logger.info("Continuous mode: recording '%s_%s' once per second.", event, direction)
        logger.info("Type 'q' then Enter to stop.\n")
        logger.info(
            "(Start your sound, then press Enter here if you want; it will keep recording "
            "regardless.)"
        )

        stop_event = threading.Event()
        t = threading.Thread(target=wait_for_quit, args=(stop_event,), daemon=True)
        t.start()

        try:
            while not stop_event.is_set():
                raw = record_1s_raw_int32_stereo()
                x = int32_to_float32_unit(raw)
                if args.normalize:
                    x = shared_rms_normalize(x)

                fname = f"{timestamp()}_{event}_{direction}.npy"
                fpath = clips_dir / fname
                np.save(fpath, x.astype(np.float32))

                append_manifest(manifest, fpath, event, direction, args.normalize)

                peak = float(np.max(np.abs(x)))
                saved += 1
                logger.info("[%s] Saved %s shape=%s peak=%.3f", saved, fname, x.shape, peak)

                if args.count and saved >= args.count:
                    break
        except KeyboardInterrupt:
            logger.info("Stopped.")

        stop_event.set()
        logger.info("Done. Total clips saved: %s", saved)
        logger.info("Manifest: %s", manifest)
        return

    try:
        while True:
            cmd = input("label> ").strip().lower()
            if not cmd:
                continue
            if cmd == "q":
                break
            if cmd not in COMMANDS:
                logger.warning("Unknown label. Valid: %s", ", ".join(list(COMMANDS.keys()) + ["q"]))
                continue

            event, direction = COMMANDS[cmd]

            raw = record_1s_raw_int32_stereo()
            x = int32_to_float32_unit(raw)
            if args.normalize:
                x = shared_rms_normalize(x)

            fname = f"{timestamp()}_{event}_{direction}.npy"
            fpath = clips_dir / fname
            np.save(fpath, x.astype(np.float32))  # (16000,2)

            append_manifest(manifest, fpath, event, direction, args.normalize)

            peak = float(np.max(np.abs(x)))
            logger.info("Saved %s shape=%s peak=%.3f", fname, x.shape, peak)

            saved += 1
            if args.count and saved >= args.count:
                break
    except KeyboardInterrupt:
        logger.info("Stopped.")

    logger.info("Done. Total clips saved: %s", saved)
    logger.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
