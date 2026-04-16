"""Compress 3DGS PLY in stages, saving viewable PLY at each step."""
import sys
import os
import numpy as np
import gzip
import struct
from plyfile import PlyData, PlyElement

def load_ply(path):
    ply = PlyData.read(path)
    v = ply.elements[0]
    N = v.count

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)

    # Count f_rest columns
    rest_names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")])
    n_rest = len(rest_names)
    if n_rest > 0:
        f_rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    opacity = np.array(v["opacity"])
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)

    sh_degree = int(((n_rest // 3) + 1) ** 0.5) - 1 if n_rest > 0 else 0

    return {
        "xyz": xyz.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
        "f_rest": f_rest.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rots": rots.astype(np.float32),
        "sh_degree": sh_degree,
        "n_rest": n_rest,
    }

def save_ply(data, path):
    """Save standard 3DGS PLY format."""
    N = data["xyz"].shape[0]
    n_rest = data["f_rest"].shape[1] if data["f_rest"].ndim == 2 else 0

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(n_rest):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    arr = np.empty(N, dtype=dtype)
    arr["x"] = data["xyz"][:, 0]
    arr["y"] = data["xyz"][:, 1]
    arr["z"] = data["xyz"][:, 2]
    arr["f_dc_0"] = data["f_dc"][:, 0]
    arr["f_dc_1"] = data["f_dc"][:, 1]
    arr["f_dc_2"] = data["f_dc"][:, 2]
    for i in range(n_rest):
        arr[f"f_rest_{i}"] = data["f_rest"][:, i]
    arr["opacity"] = data["opacity"]
    arr["scale_0"] = data["scales"][:, 0]
    arr["scale_1"] = data["scales"][:, 1]
    arr["scale_2"] = data["scales"][:, 2]
    arr["rot_0"] = data["rots"][:, 0]
    arr["rot_1"] = data["rots"][:, 1]
    arr["rot_2"] = data["rots"][:, 2]
    arr["rot_3"] = data["rots"][:, 3]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(path)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prune(data, opacity_thresh=0.005):
    """Remove Gaussians with low opacity."""
    opacities_activated = sigmoid(data["opacity"])
    mask = opacities_activated > opacity_thresh
    return {
        "xyz": data["xyz"][mask],
        "f_dc": data["f_dc"][mask],
        "f_rest": data["f_rest"][mask],
        "opacity": data["opacity"][mask],
        "scales": data["scales"][mask],
        "rots": data["rots"][mask],
        "sh_degree": data["sh_degree"],
        "n_rest": data["n_rest"],
    }

def reduce_sh(data, target_degree):
    """Reduce SH degree. degree 0 = DC only, degree 1 = 9 coeffs, etc."""
    current_degree = data["sh_degree"]
    if target_degree >= current_degree:
        return data

    n_coeffs_per_channel = (target_degree + 1) ** 2 - 1  # excluding DC
    n_rest_new = n_coeffs_per_channel * 3

    # f_rest is stored as [R0,R1,...,G0,G1,...,B0,B1,...] interleaved per channel
    # Actually in standard PLY it's stored channel-first: [C0_R, C0_G, C0_B, C1_R, ...]
    # Wait - need to check the actual ordering
    # Standard 3DGS PLY: f_rest_0..f_rest_44 where layout is [3, (degree+1)^2-1]
    # i.e., f_rest_0 = SH_1 for R, f_rest_1 = SH_1 for G, f_rest_2 = SH_1 for B, ...
    # Actually the standard ordering is: channel-first, so f_rest stores [3 x K] flattened
    # f_rest_0..f_rest_14 = R channel (15 coeffs for degree 3)
    # f_rest_15..f_rest_29 = G channel
    # f_rest_30..f_rest_44 = B channel

    current_coeffs = (current_degree + 1) ** 2 - 1  # 15 for degree 3

    if n_rest_new == 0:
        f_rest_new = np.zeros((data["xyz"].shape[0], 0), dtype=np.float32)
    else:
        # Take first n_coeffs_per_channel from each channel
        f_rest = data["f_rest"]  # [N, 3*current_coeffs]
        R = f_rest[:, :current_coeffs][:, :n_coeffs_per_channel]
        G = f_rest[:, current_coeffs:2*current_coeffs][:, :n_coeffs_per_channel]
        B = f_rest[:, 2*current_coeffs:3*current_coeffs][:, :n_coeffs_per_channel]
        f_rest_new = np.concatenate([R, G, B], axis=1)

    return {
        "xyz": data["xyz"],
        "f_dc": data["f_dc"],
        "f_rest": f_rest_new,
        "opacity": data["opacity"],
        "scales": data["scales"],
        "rots": data["rots"],
        "sh_degree": target_degree,
        "n_rest": n_rest_new,
    }

def quantize_f16(data):
    """Convert all float32 to float16 and back (simulates f16 precision loss)."""
    return {
        "xyz": data["xyz"].astype(np.float16).astype(np.float32),
        "f_dc": data["f_dc"].astype(np.float16).astype(np.float32),
        "f_rest": data["f_rest"].astype(np.float16).astype(np.float32),
        "opacity": data["opacity"].astype(np.float16).astype(np.float32),
        "scales": data["scales"].astype(np.float16).astype(np.float32),
        "rots": data["rots"].astype(np.float16).astype(np.float32),
        "sh_degree": data["sh_degree"],
        "n_rest": data["n_rest"],
    }

def save_f16_binary(data, path):
    """Save as compact float16 binary + gzip."""
    N = data["xyz"].shape[0]
    n_rest = data["f_rest"].shape[1]

    # Header: magic, N, n_rest
    header = struct.pack("<III", 0x53504C54, N, n_rest)  # "SPLT"

    # Pack all data as float16
    buf = bytearray(header)
    for arr in [data["xyz"], data["f_dc"], data["f_rest"],
                data["opacity"].reshape(-1, 1), data["scales"], data["rots"]]:
        buf += arr.astype(np.float16).tobytes()

    # Save raw
    raw_path = path + ".bin"
    with open(raw_path, "wb") as f:
        f.write(buf)

    # Save gzipped
    gz_path = path + ".bin.gz"
    with gzip.open(gz_path, "wb", compresslevel=9) as f:
        f.write(buf)

    return raw_path, gz_path

def report(name, data, path):
    N = data["xyz"].shape[0]
    sz = os.path.getsize(path)
    n_rest = data["f_rest"].shape[1]
    sh_deg = data["sh_degree"]
    print(f"  {name}: {N:,} GS, SH deg={sh_deg}, f_rest={n_rest}, file={sz/1e6:.1f}MB → {path}")

def main():
    ply_path = sys.argv[1]
    out_dir = sys.argv[2]
    scene_name = sys.argv[3]

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Compressing: {scene_name}")
    print(f"{'='*70}")

    # Stage 0: Original
    print("\n[Stage 0] Original")
    data = load_ply(ply_path)
    out0 = os.path.join(out_dir, f"{scene_name}_0_original.ply")
    save_ply(data, out0)
    report("original", data, out0)

    # Stage 1: Pruned
    print("\n[Stage 1] Pruning (opacity > 0.005)")
    data_pruned = prune(data, opacity_thresh=0.005)
    out1 = os.path.join(out_dir, f"{scene_name}_1_pruned.ply")
    save_ply(data_pruned, out1)
    report("pruned", data_pruned, out1)

    # Stage 2a: SH degree 1
    print("\n[Stage 2a] SH degree 3 → 1")
    data_sh1 = reduce_sh(data_pruned, 1)
    out2a = os.path.join(out_dir, f"{scene_name}_2a_sh1.ply")
    save_ply(data_sh1, out2a)
    report("SH deg1", data_sh1, out2a)

    # Stage 2b: SH degree 0 (DC only)
    print("\n[Stage 2b] SH degree 3 → 0 (DC only)")
    data_sh0 = reduce_sh(data_pruned, 0)
    out2b = os.path.join(out_dir, f"{scene_name}_2b_sh0.ply")
    save_ply(data_sh0, out2b)
    report("SH deg0", data_sh0, out2b)

    # Stage 3a: SH1 + float16
    print("\n[Stage 3a] SH deg1 + float16")
    data_sh1_f16 = quantize_f16(data_sh1)
    out3a = os.path.join(out_dir, f"{scene_name}_3a_sh1_f16.ply")
    save_ply(data_sh1_f16, out3a)
    report("SH1+f16", data_sh1_f16, out3a)

    # Stage 3b: SH0 + float16
    print("\n[Stage 3b] SH deg0 + float16")
    data_sh0_f16 = quantize_f16(data_sh0)
    out3b = os.path.join(out_dir, f"{scene_name}_3b_sh0_f16.ply")
    save_ply(data_sh0_f16, out3b)
    report("SH0+f16", data_sh0_f16, out3b)

    # Stage 4: Binary + gzip (for transmission size reference)
    print("\n[Stage 4] Binary f16 + gzip (transmission size)")
    raw3a, gz3a = save_f16_binary(data_sh1_f16, os.path.join(out_dir, f"{scene_name}_4a_sh1_f16"))
    raw3b, gz3b = save_f16_binary(data_sh0_f16, os.path.join(out_dir, f"{scene_name}_4b_sh0_f16"))
    print(f"  SH1+f16 bin: {os.path.getsize(raw3a)/1e6:.1f}MB, gzip: {os.path.getsize(gz3a)/1e6:.1f}MB")
    print(f"  SH0+f16 bin: {os.path.getsize(raw3b)/1e6:.1f}MB, gzip: {os.path.getsize(gz3b)/1e6:.1f}MB")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {scene_name}")
    print(f"{'='*70}")
    sizes = [
        ("0_original", out0),
        ("1_pruned", out1),
        ("2a_sh1", out2a),
        ("2b_sh0", out2b),
        ("3a_sh1_f16", out3a),
        ("3b_sh0_f16", out3b),
        ("4a_sh1_f16_gz", gz3a),
        ("4b_sh0_f16_gz", gz3b),
    ]
    orig_sz = os.path.getsize(out0)
    for name, path in sizes:
        sz = os.path.getsize(path)
        ratio = sz / orig_sz * 100
        print(f"  {name:<20} {sz/1e6:>8.1f} MB  ({ratio:>5.1f}%)")

if __name__ == "__main__":
    main()
