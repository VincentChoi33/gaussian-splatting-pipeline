"""Step 2: Structure-from-Motion with LightGlue + COLMAP."""
import logging
import shutil
import struct
from pathlib import Path

log = logging.getLogger("pipeline.sfm")


def find_largest_model(sparse_dir: Path) -> Path:
    """Find COLMAP sub-model with most registered images."""
    models = sorted(sparse_dir.iterdir())
    if not models:
        raise FileNotFoundError(f"No models in {sparse_dir}")

    best, best_count = None, 0
    for m in models:
        images_bin = m / "images.bin"
        if not images_bin.exists():
            continue
        with open(images_bin, "rb") as f:
            count = struct.unpack("<Q", f.read(8))[0]
        log.info(f"  Model {m.name}: {count} images")
        if count > best_count:
            best, best_count = m, count

    if best is None:
        raise FileNotFoundError(f"No valid models in {sparse_dir}")

    log.info(f"Selected model {best.name} ({best_count} images)")
    return best


def convert_cameras_to_pinhole(sparse_dir: Path):
    """Convert SIMPLE_RADIAL/SIMPLE_PINHOLE cameras to PINHOLE in cameras.bin."""
    cameras_bin = sparse_dir / "cameras.bin"
    if not cameras_bin.exists():
        raise FileNotFoundError(f"No cameras.bin in {sparse_dir}")

    PARAM_COUNTS = {0: 3, 1: 4, 2: 4, 3: 5}

    with open(cameras_bin, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        cameras = []
        for _ in range(num_cameras):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = PARAM_COUNTS.get(model_id)
            if n_params is None:
                raise ValueError(f"Unsupported camera model {model_id}")
            params = struct.unpack(f"<{n_params}d", f.read(n_params * 8))

            if model_id == 0:  # SIMPLE_PINHOLE
                new_params = (params[0], params[0], params[1], params[2])
                log.info(f"  Camera {cam_id}: SIMPLE_PINHOLE → PINHOLE")
            elif model_id == 2:  # SIMPLE_RADIAL
                new_params = (params[0], params[0], params[1], params[2])
                log.info(f"  Camera {cam_id}: SIMPLE_RADIAL (k={params[3]:.6f}) → PINHOLE")
            elif model_id == 3:  # RADIAL
                new_params = (params[0], params[0], params[1], params[2])
                log.info(f"  Camera {cam_id}: RADIAL → PINHOLE")
            elif model_id == 1:  # Already PINHOLE
                new_params = params
                log.info(f"  Camera {cam_id}: already PINHOLE")
            cameras.append((cam_id, 1, width, height, new_params))

    shutil.copy2(cameras_bin, str(cameras_bin) + ".bak")
    with open(cameras_bin, "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for cam_id, model_id, width, height, params in cameras:
            f.write(struct.pack("<i", cam_id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", width))
            f.write(struct.pack("<Q", height))
            f.write(struct.pack(f"<{len(params)}d", *params))

    log.info(f"Converted {num_cameras} cameras to PINHOLE")


def run_sfm(images_dir: Path, output_dir: Path, cfg: dict):
    """Run SfM pipeline: feature extraction → matching → reconstruction."""
    import pycolmap
    from hloc import (
        extract_features,
        match_features,
        reconstruction,
    )

    sfm_dir = output_dir / "sparse"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs[cfg.get("feature", "superpoint")]
    matcher_conf = match_features.confs[
        "lightglue" if cfg.get("matcher") == "lightglue" else "superglue"
    ]

    sfm_temp = sfm_dir / "sfm_temp"

    log.info(f"Running SfM: {images_dir}")
    log.info(f"  Feature: {cfg.get('feature', 'superpoint')}, Matcher: {cfg.get('matcher', 'lightglue')}")

    feature_path = extract_features.main(feature_conf, images_dir, output_dir)
    match_path = match_features.main(matcher_conf, feature_path, output_dir=output_dir)

    reconstruction.main(
        sfm_dir=sfm_temp,
        image_dir=images_dir,
        pairs=match_path,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    best_model = find_largest_model(sfm_temp)
    target = sfm_dir / "0"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(best_model, target)
    shutil.rmtree(sfm_temp)

    log.info(f"SfM complete → {target}")

    if cfg.get("convert_to_pinhole", True):
        convert_cameras_to_pinhole(target)
