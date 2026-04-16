import pytest
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def _make_test_ply(path: Path, n: int = 1000):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(45):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    arr = np.zeros(n, dtype=dtype)
    arr["x"] = np.random.randn(n)
    arr["y"] = np.random.randn(n)
    arr["z"] = np.random.randn(n)
    arr["f_dc_0"] = np.random.randn(n)
    arr["f_dc_1"] = np.random.randn(n)
    arr["f_dc_2"] = np.random.randn(n)
    arr["opacity"] = np.random.randn(n)
    arr["scale_0"] = np.random.randn(n)
    arr["scale_1"] = np.random.randn(n)
    arr["scale_2"] = np.random.randn(n)
    arr["rot_0"] = 1.0
    arr["rot_1"] = 0.0
    arr["rot_2"] = 0.0
    arr["rot_3"] = 0.0

    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path


def test_compress_default(tmp_path):
    from pipeline.compress import run_compress

    ply = _make_test_ply(tmp_path / "full.ply", n=1000)
    cfg = {"sh_degree": 0, "float16": True, "prune_threshold": 0.005, "downsample": 0.5}
    run_compress(ply, tmp_path, cfg)

    assert (tmp_path / "compressed.ply").exists()
    assert (tmp_path / "compressed_ds50.ply").exists()

    assert (tmp_path / "compressed.ply").stat().st_size < ply.stat().st_size
    assert (tmp_path / "compressed_ds50.ply").stat().st_size < (tmp_path / "compressed.ply").stat().st_size
