"""Export full 3DGS PLY from lightning checkpoint."""
import sys
import os
import torch

sys.path.insert(0, "/data/choihy/GSplat/gaussian-splatting-lightning")

from internal.utils.gaussian_utils import GaussianPlyUtils

def export(ckpt_path, output_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]

    # Extract gaussian model parameters
    prefix = "gaussian_model.gaussians."
    keys = [k for k in state if k.startswith(prefix)]
    print(f"Found {len(keys)} gaussian keys: {[k.replace(prefix,'') for k in keys]}")

    xyz = state[f"{prefix}means"].numpy()           # [N, 3]
    shs_dc = state[f"{prefix}shs_dc"].numpy()       # [N, 1, 3]
    shs_rest = state[f"{prefix}shs_rest"].numpy()    # [N, 15, 3]
    opacities = state[f"{prefix}opacities"].numpy()  # [N, 1]
    scales = state[f"{prefix}scales"].numpy()        # [N, 3]
    rotations = state[f"{prefix}rotations"].numpy()  # [N, 4]

    n = xyz.shape[0]
    sh_degrees = int((shs_rest.shape[1] + 1) ** 0.5) - 1 if shs_rest.shape[1] > 0 else 0
    print(f"Gaussians: {n}, SH degree: {sh_degrees}")

    # GaussianPlyUtils expects features_dc [N, C, 1] and features_rest [N, C, K]
    # shs_dc is [N, 1, 3] → transpose to [N, 3, 1]
    # shs_rest is [N, 15, 3] → transpose to [N, 3, 15]
    ply_utils = GaussianPlyUtils(
        sh_degrees=sh_degrees,
        xyz=xyz,
        opacities=opacities,
        features_dc=shs_dc.transpose(0, 2, 1),    # [N, 3, 1]
        features_rest=shs_rest.transpose(0, 2, 1), # [N, 3, 15]
        scales=scales,
        rotations=rotations,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ply_utils.save_to_ply(output_path)
    print(f"Saved to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")

if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    output_path = sys.argv[2]
    export(ckpt_path, output_path)
