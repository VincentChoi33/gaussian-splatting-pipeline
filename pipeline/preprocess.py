"""Step 1: Video frame extraction and blur filtering."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("pipeline.preprocess")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def compute_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path: Path, output_dir: Path, fps: int = 2) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "1",
        str(output_dir / "frame_%04d.jpg"),
        "-y",
    ]
    log.info(f"Extracting frames at {fps} fps: {video_path}")
    subprocess.run(cmd, check=True, capture_output=True)
    n = len(list(output_dir.glob("frame_*.jpg")))
    log.info(f"Extracted {n} frames → {output_dir}")
    return output_dir


def filter_blurry_frames(images_dir: Path, percentile: int = 20) -> list[Path]:
    files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No images in {images_dir}")

    scores = []
    for f in files:
        img = cv2.imread(str(f))
        scores.append((f, compute_blur_score(img)))

    threshold = np.percentile([s for _, s in scores], percentile)
    kept = [f for f, s in scores if s >= threshold]
    removed = len(files) - len(kept)

    for f, s in scores:
        if s < threshold:
            f.unlink()

    log.info(f"Blur filter: {len(files)} → {len(kept)} frames ({removed} removed, threshold={threshold:.1f})")
    return kept


def run_preprocess(input_path: Path, output_dir: Path, cfg: dict) -> Path:
    images_dir = output_dir / "images"

    if is_video(input_path):
        extract_frames(input_path, images_dir, fps=cfg.get("fps", 2))
        if cfg.get("blur_filter", True):
            filter_blurry_frames(images_dir, percentile=cfg.get("blur_percentile", 20))
    else:
        if input_path != images_dir:
            images_dir.mkdir(parents=True, exist_ok=True)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for f in input_path.glob(ext):
                    shutil.copy2(f, images_dir / f.name)
        n = len(list(images_dir.iterdir()))
        log.info(f"Copied {n} images → {images_dir}")

    return images_dir
