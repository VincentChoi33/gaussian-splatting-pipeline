"""Downsample Gaussians by importance, save viewable PLY at each level."""
import sys
import os
import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def load_ply(path):
    ply = PlyData.read(path)
    v = ply.elements[0]
    N = v.count
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rest_names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")])
    n_rest = len(rest_names)
    f_rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32)
    opacity = np.array(v["opacity"], dtype=np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    return xyz, f_dc, f_rest, opacity, scales, rots, n_rest


def save_ply(path, xyz, f_dc, f_rest, opacity, scales, rots):
    N = xyz.shape[0]
    n_rest = f_rest.shape[1]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(n_rest):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.empty(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(n_rest):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacity
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def compute_importance(opacity, scales):
    """Importance = opacity × volume. Big visible Gaussians matter most."""
    opacity_act = sigmoid(opacity)
    volume = np.exp(scales).prod(axis=1)
    return opacity_act * volume


def main():
    ply_path = sys.argv[1]
    out_dir = sys.argv[2]
    name = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    xyz, f_dc, f_rest, opacity, scales, rots, n_rest = load_ply(ply_path)
    N = xyz.shape[0]
    orig_sz = os.path.getsize(ply_path)

    # Compute importance and sort
    importance = compute_importance(opacity, scales)
    order = np.argsort(-importance)  # most important first

    print(f"\n{'='*60}")
    print(f"{name}: {N:,} Gaussians, {orig_sz/1e6:.1f}MB")
    print(f"{'='*60}")
    print(f"{'비율':<8} {'Gaussian수':>12} {'파일크기':>10} {'원본대비':>8}")
    print("-" * 60)

    for pct in [100, 75, 50, 25, 10, 5]:
        k = int(N * pct / 100)
        idx = order[:k]

        out_path = os.path.join(out_dir, f"{name}_{pct}pct.ply")
        save_ply(out_path, xyz[idx], f_dc[idx], f_rest[idx],
                 opacity[idx], scales[idx], rots[idx])

        sz = os.path.getsize(out_path)
        print(f"{pct:>5}%  {k:>12,}  {sz/1e6:>8.1f}MB  {sz/orig_sz*100:>7.1f}%")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
