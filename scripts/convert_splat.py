"""Convert 3DGS PLY to .splat format (compact, widely supported by web viewers).

.splat format per Gaussian (32 bytes total):
  - position: 3 x float32 (12 bytes)
  - scale:    3 x float32 (12 bytes) -- exp(log_scale)
  - color:    4 x uint8   (4 bytes)  -- RGBA, RGB from SH0, A from sigmoid(opacity)
  - rotation: 4 x uint8   (4 bytes)  -- quaternion mapped to [0, 255]
"""
import sys
import os
import struct
import numpy as np
from plyfile import PlyData


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def load_and_convert(ply_path, output_path, prune_thresh=0.005):
    ply = PlyData.read(ply_path)
    v = ply.elements[0]
    N = v.count

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    opacity_logit = np.array(v["opacity"], dtype=np.float32)
    scales_log = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    # Prune
    opacity_act = sigmoid(opacity_logit)
    mask = opacity_act > prune_thresh
    xyz = xyz[mask]
    f_dc = f_dc[mask]
    opacity_act = opacity_act[mask]
    scales_log = scales_log[mask]
    rots = rots[mask]
    N = xyz.shape[0]
    print(f"After pruning: {N:,} Gaussians")

    # Convert SH DC to RGB [0, 255]
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0, 1)
    rgb_u8 = (rgb * 255).astype(np.uint8)
    alpha_u8 = (opacity_act * 255).astype(np.uint8)
    rgba = np.column_stack([rgb_u8, alpha_u8])  # [N, 4]

    # Activated scales
    scales = np.exp(scales_log).astype(np.float32)

    # Normalize quaternions and map to uint8 [0, 255]
    # q in [-1, 1] → uint8: (q * 128 + 128)
    norms = np.linalg.norm(rots, axis=1, keepdims=True)
    rots_norm = rots / np.maximum(norms, 1e-8)
    rot_u8 = np.clip(rots_norm * 128 + 128, 0, 255).astype(np.uint8)

    # Sort by size (largest first) for better rendering
    sizes = np.prod(scales, axis=1)
    order = np.argsort(-sizes)

    # Write .splat binary
    with open(output_path, "wb") as f:
        for i in order:
            f.write(struct.pack("<3f", *xyz[i]))
            f.write(struct.pack("<3f", *scales[i]))
            f.write(rgba[i].tobytes())
            f.write(rot_u8[i].tobytes())

    sz = os.path.getsize(output_path)
    print(f"Saved: {output_path} ({sz / 1e6:.1f} MB, {N:,} GS, {sz/N:.0f} bytes/GS)")
    return N, sz


def main():
    ply_path = sys.argv[1]
    out_dir = sys.argv[2]
    name = sys.argv[3]

    os.makedirs(out_dir, exist_ok=True)

    print(f"\nConverting {name}...")
    out_path = os.path.join(out_dir, f"{name}.splat")
    n, sz = load_and_convert(ply_path, out_path)

    orig_sz = os.path.getsize(ply_path)
    print(f"\n원본 PLY: {orig_sz/1e6:.1f} MB → .splat: {sz/1e6:.1f} MB ({sz/orig_sz*100:.1f}%)")


if __name__ == "__main__":
    main()
