import os
import torch
import numpy as np

import pc_tools  # Make sure pc_tools has random_spatial_crop_pc, etc.

def process_point_clouds(
    batched_pcs,
    original_name="cloud",   # <--- Use this for file naming
    noise_std_min=0.001,
    noise_std_max=0.01,
    fraction=0.5,          # fraction to remove for random spatial crop
    downsample_ratio=0.5,
    seed=42,
    output_dir="./processed_pcds",
    visualize=False
):
    """
    Process batched point clouds to create:
        1) Original (for visualization only)
        2) Noisy
        3) Incomplete (random spatial crop)
        4) Sparse (random downsample)
        5) Combined (Noisy -> Crop -> Downsample)

    All results are saved with the filename:
        <original_name>.pcd
    in each respective subfolder (noisy, incomplete, sparse, combined).

    Args:
        batched_pcs: torch.Tensor or np.array of shape (B, N, 3) or (B, N, 6)
        original_name: str, used for saving .pcd (e.g., "02691156-1a04e3eab45ca15dd86060f189eb133")
        noise_std_min: float, min Gaussian noise std
        noise_std_max: float, max Gaussian noise std
        fraction: float, fraction of points to remove (random plane)
        downsample_ratio: float in (0,1], ratio of points to keep for random downsampling
        seed: int, random seed for reproducibility
        output_dir: str, parent directory to save the processed pcd files
        visualize: bool, if True, opens visualization windows for each step
    """

    # Convert to NumPy if needed
    if isinstance(batched_pcs, torch.Tensor):
        batched_pcs = batched_pcs.cpu().numpy()

    B = batched_pcs.shape[0]  # batch size

    # Create output subfolders
    noisy_dir = os.path.join(output_dir, "Denoising")
    incomplete_dir = os.path.join(output_dir, "Completion")
    sparse_dir = os.path.join(output_dir, "Upsampling")
    combined_dir = os.path.join(output_dir, "Combination")

    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(incomplete_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for b_idx in range(B):
        pc_data = batched_pcs[b_idx]  # (N,3) or (N,6)

        # --- Original Visualization ---
        if visualize:
            print(f"\n[{b_idx}] Visualizing ORIGINAL point cloud...")
            pc_tools.show_point_cloud(pc_data, point_size=0.02)

        # --- (1) Noisy ---
        noisy_pc, _ = pc_tools.add_GaussianNoise(noise_std_min, noise_std_max, pc_data)
        if visualize:
            print(f"[{b_idx}] Visualizing NOISY point cloud...")
            pc_tools.show_point_cloud(noisy_pc, point_size=0.02)

        # --- (2) Incomplete: Random Spatial Crop ---
        incomplete_pc = pc_tools.random_spatial_crop_pc(
            pc_data, fraction=fraction, seed=seed
        )
        if visualize:
            print(f"[{b_idx}] Visualizing INCOMPLETE (random spatial crop) point cloud...")
            pc_tools.show_point_cloud(incomplete_pc, point_size=0.02)

        # --- (3) Sparse: Random Downsample ---
        sparse_pc = pc_tools.random_downsample_pc(
            pc_data, keep_ratio=downsample_ratio, seed=seed
        )
        if visualize:
            print(f"[{b_idx}] Visualizing SPARSE (random downsample) point cloud...")
            pc_tools.show_point_cloud(sparse_pc, point_size=0.02)

        # --- (4) Combined ---
        combined_pc, _ = pc_tools.add_GaussianNoise(noise_std_min, noise_std_max, pc_data)
        combined_pc = pc_tools.random_spatial_crop_pc(combined_pc, fraction=fraction, seed=seed)
        combined_pc = pc_tools.random_downsample_pc(combined_pc, keep_ratio=downsample_ratio, seed=seed)
        if visualize:
            print(f"[{b_idx}] Visualizing COMBINED (noisy + crop + downsample) point cloud...")
            pc_tools.show_point_cloud(combined_pc, point_size=0.02)

        # --- Save results as PCD files ---
        # Use the original_name to name the .pcd
        pc_tools.save_pcd(noisy_pc,      os.path.join(noisy_dir,      f"{original_name}.pcd"))
        pc_tools.save_pcd(incomplete_pc, os.path.join(incomplete_dir, f"{original_name}.pcd"))
        pc_tools.save_pcd(sparse_pc,     os.path.join(sparse_dir,     f"{original_name}.pcd"))
        pc_tools.save_pcd(combined_pc,   os.path.join(combined_dir,   f"{original_name}.pcd"))


def main():
    # Suppose we have a single .npy file:
    input_path = "Datasets/KITTI-360/02691156-1a04e3eab45ca15dd86060f189eb133.npy"

    # Extract the original filename without extension (e.g., "02691156-1a04e3eab45ca15dd86060f189eb133")
    import os
    base_name = os.path.basename(input_path)                  # "02691156-1a04e3eab45ca15dd86060f189eb133.npy"
    original_stem = os.path.splitext(base_name)[0]            # "02691156-1a04e3eab45ca15dd86060f189eb133"

    # Load the .npy as a NumPy array
    pc_data = np.load(input_path)
    print("Loaded pc_data shape:", pc_data.shape)  # (N, 3) or (N, 6)

    # Expand dims for batch dimension => (1, N, D)
    if len(pc_data.shape) == 2:
        pc_data = np.expand_dims(pc_data, axis=0)

    # Run the process
    process_point_clouds(
        batched_pcs=pc_data,
        original_name=original_stem,   # <--- pass the stem to use for saving
        noise_std_min=0.001,
        noise_std_max=0.01,
        fraction=0.5,           # remove fraction of the points
        downsample_ratio=0.25,  # keep downsample_ratio of points for the sparse
        seed=42,
        output_dir="./processed_pcds",
        visualize=True
    )

if __name__ == "__main__":
    main()
