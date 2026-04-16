"""Compress 3DGS keeping ALL SH data. Various strategies."""
import sys
import os
import struct
import gzip
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


def prune(xyz, f_dc, f_rest, opacity, scales, rots, thresh=0.005):
    mask = sigmoid(opacity) > thresh
    return xyz[mask], f_dc[mask], f_rest[mask], opacity[mask], scales[mask], rots[mask]


def save_ply_f32(path, xyz, f_dc, f_rest, opacity, scales, rots):
    """Standard float32 PLY."""
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
    PlyElement.describe(arr, "vertex")
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def quant_to_int16(arr):
    """Quantize float array to int16 with per-column min/max."""
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (arr - mins) / ranges  # [0, 1]
    q = (normalized * 65535).astype(np.uint16)
    return q, mins.astype(np.float32), ranges.astype(np.float32)


def dequant_int16(q, mins, ranges):
    return (q.astype(np.float32) / 65535.0) * ranges + mins


def quant_to_uint8(arr):
    """Quantize float array to uint8 with per-column min/max."""
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (arr - mins) / ranges
    q = (normalized * 255).astype(np.uint8)
    return q, mins.astype(np.float32), ranges.astype(np.float32)


def write_compact(path, xyz, f_dc, f_rest, opacity, scales, rots, dtype="f16"):
    """
    Custom binary format:
    Header: magic(4) + N(4) + n_rest(4) + dtype_flag(4)
    Then per-attribute: [min/range metadata if quantized] + [data]
    """
    N = xyz.shape[0]
    n_rest = f_rest.shape[1]
    buf = bytearray()

    if dtype == "f16":
        # Everything as float16
        flag = 1
        buf += struct.pack("<IIII", 0x33444753, N, n_rest, flag)  # "3DGS"
        buf += xyz.astype(np.float16).tobytes()
        buf += f_dc.astype(np.float16).tobytes()
        buf += f_rest.astype(np.float16).tobytes()
        buf += opacity.astype(np.float16).tobytes()
        buf += scales.astype(np.float16).tobytes()
        buf += rots.astype(np.float16).tobytes()

    elif dtype == "i16":
        # xyz as float32 (precision matters), rest as int16 quantized
        flag = 2
        buf += struct.pack("<IIII", 0x33444753, N, n_rest, flag)
        buf += xyz.astype(np.float32).tobytes()
        for arr in [f_dc, f_rest, opacity.reshape(-1, 1), scales, rots]:
            q, mins, ranges = quant_to_int16(arr)
            buf += mins.tobytes()
            buf += ranges.tobytes()
            buf += q.tobytes()

    elif dtype == "mixed":
        # xyz: f32, f_dc: f16, f_rest: uint8(!), opacity+scale+rot: f16
        flag = 3
        buf += struct.pack("<IIII", 0x33444753, N, n_rest, flag)
        buf += xyz.astype(np.float32).tobytes()
        buf += f_dc.astype(np.float16).tobytes()
        # f_rest as uint8 with per-column min/range
        q, mins, ranges = quant_to_uint8(f_rest)
        buf += mins.tobytes()
        buf += ranges.tobytes()
        buf += q.tobytes()
        buf += opacity.astype(np.float16).tobytes()
        buf += scales.astype(np.float16).tobytes()
        buf += rots.astype(np.float16).tobytes()

    with open(path, "wb") as f:
        f.write(buf)

    # Also gzip
    gz_path = path + ".gz"
    with gzip.open(gz_path, "wb", compresslevel=9) as f:
        f.write(buf)

    return os.path.getsize(path), os.path.getsize(gz_path)


def morton_sort(xyz):
    """Sort by Morton code (Z-order curve) for spatial coherence."""
    # Normalize to [0, 1023]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    norm = ((xyz - mins) / ranges * 1023).astype(np.uint32)

    # Interleave bits for 3D Morton code (simplified)
    def spread(v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    codes = spread(norm[:, 0]) | (spread(norm[:, 1]) << 1) | (spread(norm[:, 2]) << 2)
    return np.argsort(codes)


def main():
    ply_path = sys.argv[1]
    out_dir = sys.argv[2]
    name = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Compressing (keeping ALL SH): {name}")
    print(f"{'='*70}")

    xyz, f_dc, f_rest, opacity, scales, rots, n_rest = load_ply(ply_path)
    N_orig = xyz.shape[0]
    orig_sz = os.path.getsize(ply_path)
    print(f"원본: {N_orig:,} GS, {n_rest} f_rest, {orig_sz/1e6:.1f}MB")

    # Prune
    xyz, f_dc, f_rest, opacity, scales, rots = prune(xyz, f_dc, f_rest, opacity, scales, rots)
    N = xyz.shape[0]
    print(f"Pruned: {N:,} GS ({N_orig - N:,} 제거)")

    # Sort by Morton code for better compression
    order = morton_sort(xyz)
    xyz, f_dc, f_rest = xyz[order], f_dc[order], f_rest[order]
    opacity, scales, rots = opacity[order], scales[order], rots[order]

    results = []

    # 0. Baseline: pruned PLY (f32)
    p = os.path.join(out_dir, f"{name}_baseline.ply")
    save_ply_f32(p, xyz, f_dc, f_rest, opacity, scales, rots)
    sz = os.path.getsize(p)
    results.append(("baseline (pruned f32 PLY)", sz, "-", p))

    # 1. PLY + gzip
    gz_p = p + ".gz"
    with open(p, "rb") as f_in:
        with gzip.open(gz_p, "wb", compresslevel=9) as f_out:
            f_out.write(f_in.read())
    gz_sz = os.path.getsize(gz_p)
    results.append(("PLY + gzip", gz_sz, "-", gz_p))

    # 2. Morton sorted PLY + gzip (spatial coherence helps gzip)
    p2 = os.path.join(out_dir, f"{name}_morton.ply")
    save_ply_f32(p2, xyz, f_dc, f_rest, opacity, scales, rots)
    gz_p2 = p2 + ".gz"
    with open(p2, "rb") as f_in:
        with gzip.open(gz_p2, "wb", compresslevel=9) as f_out:
            f_out.write(f_in.read())
    gz2_sz = os.path.getsize(gz_p2)
    results.append(("Morton sort + PLY + gzip", gz2_sz, "-", gz_p2))

    # 3. All float16 binary
    p3 = os.path.join(out_dir, f"{name}_f16.bin")
    raw_sz, gz_sz = write_compact(p3, xyz, f_dc, f_rest, opacity, scales, rots, "f16")
    results.append(("float16 binary", raw_sz, gz_sz, p3))

    # 4. int16 quantized (xyz f32, rest int16)
    p4 = os.path.join(out_dir, f"{name}_i16.bin")
    raw_sz, gz_sz = write_compact(p4, xyz, f_dc, f_rest, opacity, scales, rots, "i16")
    results.append(("int16 quant (xyz f32)", raw_sz, gz_sz, p4))

    # 5. Mixed: xyz f32, f_dc f16, f_rest uint8, rest f16
    p5 = os.path.join(out_dir, f"{name}_mixed.bin")
    raw_sz, gz_sz = write_compact(p5, xyz, f_dc, f_rest, opacity, scales, rots, "mixed")
    results.append(("mixed (f_rest uint8)", raw_sz, gz_sz, p5))

    # 6. Verify quality: dequantize int16 back and save PLY for viewing
    q_dc, dc_min, dc_rng = quant_to_int16(f_dc)
    q_rest, rest_min, rest_rng = quant_to_int16(f_rest)
    q_op, op_min, op_rng = quant_to_int16(opacity.reshape(-1, 1))
    q_sc, sc_min, sc_rng = quant_to_int16(scales)
    q_rot, rot_min, rot_rng = quant_to_int16(rots)
    p6 = os.path.join(out_dir, f"{name}_i16_preview.ply")
    save_ply_f32(p6,
                 xyz,
                 dequant_int16(q_dc, dc_min, dc_rng),
                 dequant_int16(q_rest, rest_min, rest_rng),
                 dequant_int16(q_op, op_min, op_rng).flatten(),
                 dequant_int16(q_sc, sc_min, sc_rng),
                 dequant_int16(q_rot, rot_min, rot_rng))
    p6_sz = os.path.getsize(p6)
    results.append(("int16 preview PLY (viewable)", p6_sz, "-", p6))

    # 7. Mixed uint8 f_rest preview PLY
    q8_rest, r8_min, r8_rng = quant_to_uint8(f_rest)
    p7 = os.path.join(out_dir, f"{name}_u8rest_preview.ply")
    save_ply_f32(p7,
                 xyz, f_dc,
                 dequant_int16(np.stack([quant_to_uint8(f_rest[:, i:i+1])[0] for i in range(f_rest.shape[1])], axis=1).reshape(f_rest.shape).astype(np.uint16) * 257,
                               *quant_to_int16(f_rest)[1:]),  # hack: just use uint8 dequant
                 opacity, scales, rots)
    # Actually let's do it properly
    rest_u8, u8_min, u8_rng = quant_to_uint8(f_rest)
    rest_dequant = (rest_u8.astype(np.float32) / 255.0) * u8_rng + u8_min
    save_ply_f32(p7, xyz, f_dc, rest_dequant, opacity, scales, rots)
    p7_sz = os.path.getsize(p7)
    results.append(("uint8 f_rest preview PLY (viewable)", p7_sz, "-", p7))

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {name} (SH 전부 유지)")
    print(f"{'='*70}")
    base_sz = results[0][1]
    print(f"{'방법':<35} {'크기':>10} {'gzip':>10} {'비율':>8}")
    print("-" * 70)
    for label, sz, gz, path in results:
        sz_mb = f"{sz/1e6:.1f}MB"
        gz_mb = f"{gz/1e6:.1f}MB" if gz != "-" else "-"
        ratio = f"{sz/orig_sz*100:.1f}%"
        print(f"{label:<35} {sz_mb:>10} {gz_mb:>10} {ratio:>8}")

    # Also show gzip versions
    print("-" * 70)
    for label, sz, gz, path in results:
        if gz != "-":
            gz_mb = f"{gz/1e6:.1f}MB"
            ratio = f"{gz/orig_sz*100:.1f}%"
            print(f"{label + ' +gz':<35} {gz_mb:>10} {'':>10} {ratio:>8}")


if __name__ == "__main__":
    main()
