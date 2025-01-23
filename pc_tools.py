"""
Yi Du's point cloud Toolbox
"""

import os
import numpy as np
import torch
import random
import open3d as o3d
import cv2

import pytorch3d.ops as ops

try:
    import pptk
except:
    print("Warning: pptk is not installed! Ignore this if you don't need to visualize the point cloud.")
    pptk = None
    pass


# ---------------------------------- Point cloud visualization ----------------------------------------

def show_point_cloud(pc_data, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=False):
    """
    Show point cloud. If the point cloud is colored (dimension is 6), the color will be shown.
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        point_size: float, size of the point
        bg_color: list, background color
        show_grid: bool, whether to show the grid
    Return:
        window: pptk.viewer object for further operations like view manipulation and image capture
    """
    if pptk is not None:
        if pc_data.shape[1] == 3:
            window = pptk.viewer(pc_data)
        elif pc_data.shape[1] == 6:
            if np.max(pc_data[:, 3:]) > 1 + 0.9:
                pc_data[:, 3:] = pc_data[:, 3:]/255 # convert rgb from 0-255 to 0-1
            window = pptk.viewer(pc_data[:, :3], pc_data[:, 3:])

        # set point size
        window.set(point_size=point_size)
        # set background color
        window.set(bg_color=bg_color)
        # remove grid
        window.set(show_grid=show_grid)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        if pc_data.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(pc_data[:, 3:6])
        # window = o3d.visualization.draw_geometries([pcd],
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
        #                                   up=[-0.0694, -0.9768, 0.2024])
        window = o3d.visualization.Visualizer()
        window.create_window(visible=True)
        # Convert rgba to rgb
        bg_color_rgb = bg_color[:3]
        # set background color
        window.get_render_option().background_color = bg_color_rgb
        window.add_geometry(pcd)
        window.run()

    return window


def display_inlier_outlier(cloud, ind):
    """
    Display inlier and outlier point clouds
    Parameter:
        cloud: open3d.geometry.PointCloud object
        ind: numpy array of inliers
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def windows_close_ctrl(windows):
    """
    Close all the pptk viewer windows
    Parameter:
        windows: list, pptk.viewer object
    """
    cv2.namedWindow('Key Listener')
    while True:
        key = cv2.waitKey(0)  # Listen for a key event
        if key == 27:  # ASCII value of 'esc' is 27
            cv2.destroyAllWindows()   
            for window in windows:
                window.close()  # Close each pptk viewer window
            break
# ---------------------------------- Point cloud visualization ----------------------------------------



# ---------------------------------- Point cloud IO ----------------------------------------
def read_SuperMap_hdf5(path):
    """
    Read hdf5 file in SuperMap format
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    import h5py
    f = h5py.File(path, 'r')
    data_train = f['02801938']['train']
    data_test = f['02801938']['test']
    data = {'train':np.array(data_train), 'test':np.array(data_test)}
    
    return data


def read_off(path):
    """
    Read off file format point cloud
    Parameter:
        file: path of off file
    Return:
        verts: numpy array of pointcloud [[x, y, z], ...]
        faces: numpy array of faces [[v1, v2, v3], ...]
    """
    with open(path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, __ = map(int, file.readline().strip().split(' '))
        verts = np.array([[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)])
        faces = [[int(s) for s in file.readline().strip().split(' ')[1:]] for _ in range(n_faces)]
    

    return verts, faces


def read_xyz(path):
    """
    Read xyz file 
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    data = np.genfromtxt(path, delimiter=' ', skip_header=0)
    

    return np.array(data)


def read_pcd(path):
    """
    Read pcd file generated by ./func.py/save_pc
    Parameter:
        path: path of pcd file
    Return:
        xyzrgb: numpy array of colored pointcloud [[x, y, z. r, g, b], ...]
    """
    xyzrgb = []
    with open(path, 'r') as f:
        content = f.readlines()
        for i in content[10:]:
            i_content = i.split(" ")
            x, y, z = float(i_content[0]), float(i_content[1]), float(i_content[2])
            r, g, b = float(i_content[3]), float(i_content[4]), float(i_content[5][:-1])

            xyzrgb.append([x,y,z,r,g,b])

    return np.array(xyzrgb)


def read_bin_pc(path, pc_show=False):
    """
    Read bin file format point cloud
    Parameter:
        path: path of pcd file
        pc_show: bool, whether to show the point cloud.
    Return:
        xyzr: numpy array of pointcloud [[x, y, z, r], ...]
    """
    point_cloud_data = np.fromfile(path, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r
    point_cloud_data = point_cloud_data[:, :3]
    if pc_show:
        show_point_cloud(pcd_down)

    return point_cloud_data


def save_pc_file(data, path):
    """
    Save point cloud file
    Parameter:
        data: numpy array of pointcloud [[x, y, z], ...]
        path: path of pcd file
    """
    np.savetxt(path, data, fmt='%.6f')
    # np.save(path, data)
    # data.tofile(path)

def save_pcd(pc_data, filename):
    """
    Save a point cloud to a .pcd file, in a format compatible with read_pcd in this file.
    The user-defined read_pcd() expects:
      - First 10 lines are PCD header
      - From line 10 onward: "x y z r g b"
    
    Args:
        pc_data: np.array of shape (N, 3) or (N, 6)
        filename: string, path to save the .pcd file
    """
    # If the point cloud has only XYZ, we will assign a default color (255, 255, 255).
    has_color = (pc_data.shape[1] == 6)

    # Prepare header lines (exactly 10 lines)
    lines = []
    lines.append("# .PCD v0.7 - Point Cloud Data file format")
    lines.append("VERSION 0.7")
    lines.append("FIELDS x y z r g b")
    lines.append("SIZE 4 4 4 4 4 4")
    lines.append("TYPE F F F F F F")
    lines.append("COUNT 1 1 1 1 1 1")
    lines.append(f"WIDTH {pc_data.shape[0]}")
    lines.append("HEIGHT 1")
    lines.append(f"POINTS {pc_data.shape[0]}")
    lines.append("DATA ascii")

    with open(filename, 'w') as f:
        # Write header
        for line in lines:
            f.write(line + "\n")

        # Write each point
        for i in range(pc_data.shape[0]):
            x = pc_data[i, 0]
            y = pc_data[i, 1]
            z = pc_data[i, 2]
            if has_color:
                r = pc_data[i, 3]
                g = pc_data[i, 4]
                b = pc_data[i, 5]
            else:
                # Default color if none is provided
                r = 255
                g = 255
                b = 255

            # Make sure we write six columns so that read_pcd can parse them
            f.write(f"{x} {y} {z} {r} {g} {b}\n")



# ---------------------------------- Point cloud IO ----------------------------------------



# ---------------------------------- Point cloud operations ----------------------------------------
def noise_Gaussian(points, mean):
    """
    Add Gaussian noise to the point cloud
    Parameter:
        points: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        mean: float, mean of the Gaussian distribution
    Return:
        pc_out: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
    """
    noise = np.random.normal(0, mean, points.shape)
    pc_out = points + noise
    return pc_out


def voxel_downsample_pc(pc_data, voxel_size=0.5, pc_show=False):
    """
    Downsample the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        voxel_size: float, size of the voxel
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_down: numpy array of downsampled pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
    """
    if pc_data.shape[1] == 6:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_data[:, 3:])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_rgb_down = downpcd.colors
        pcd_down = np.hstack((pcd_xyz_down, pcd_rgb_down))
        if pc_show:
            show_point_cloud(pcd_down)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_down = np.array(pcd_xyz_down)
        if pc_show:
            show_point_cloud(pcd_down)
    else:
        assert False, "Error: point cloud data shape is not correct!"


    return pcd_down


def uniform_downsample_pc(pc_data, every_k_points=10, pc_show=False):
    """
    Downsample the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
        every_k_points: int, every k points to keep
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_down: numpy array of downsampled pointcloud [[x, y, z], ...] or [[x, y, z, r, g, b], ...]
    """
    if pc_data.shape[1] == 6:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_data[:, 3:])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        # downpcd = pcd.voxel_down_sample(voxel_size=0.5)
        downpcd = rm.uniform_down_sample(every_k_points=every_k_points)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_rgb_down = downpcd.colors
        pcd_down = np.hstack((pcd_xyz_down, pcd_rgb_down))
        if pc_show:
            show_point_cloud(pcd_down)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data[:, :3])
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=680,std_ratio=3.8)
        # downpcd = pcd.voxel_down_sample(voxel_size=0.5)
        downpcd = rm.uniform_down_sample(every_k_points=every_k_points)
        # o3d.visualization.draw_geometries([downpcd])
        pcd_xyz_down = downpcd.points
        pcd_down = np.array(pcd_xyz_down)
        if pc_show:
            show_point_cloud(pcd_down)
    else:
        assert False, "Error: point cloud data shape is not correct!"


    return pcd_down

def random_downsample_pc(pc_data, keep_ratio=0.5, seed=None):
    """
    Randomly downsample the input point cloud.

    Args:
        pc_data: np.array of shape (N, 3) or (N, 6)
        keep_ratio: float in (0, 1], ratio of points to keep
        seed: int or None, if set, random seed for reproducibility

    Returns:
        pc_down: Downsampled np.array of shape (M, 3) or (M, 6) where M = int(N * keep_ratio)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    N = pc_data.shape[0]
    keep_num = int(N * keep_ratio)

    # Randomly choose indices
    indices = np.random.choice(N, keep_num, replace=False)
    pc_down = pc_data[indices]
    return pc_down


def divide_pc_xyz_rgb(pc_data, pc_show=False):
    """
    1. Divide the point cloud into xyz and rgb.  2. Convert rgb from 0-255 to 0-1.
    Parameter:
        The input point cloud should be in the format of [[x, y, z, r, g, b], ...]
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_xyz: numpy array of pointcloud [[x, y, z], ...]
        pcd_rgb: numpy array of pointcloud [[r, g, b], ...]
        pcd_data: numpy array of pointcloud [[x, y, z, r, g, b], ...]
    """
    assert pc_data.shape[1] == 6, "Error: point cloud data shape is not correct!"

    # devide the pointcloud into xyz and rgb
    if np.max(pc_data[:, 3:]) > 1 + 0.9:
        pc_data[:, 3:] = pc_data[:, 3:]/255 # convert rgb from 0-255 to 0-1
    pcd_xyz = pc_data[:, :3]
    pcd_rgb = pc_data[:, 3:]
    pcd_data = np.hstack((pcd_xyz, pcd_rgb))

    if pc_show:
        show_point_cloud(pcd_data)

    return pcd_xyz, pcd_rgb, pcd_data


def rm_statistical_outlier(pc_data, nb_neighbors=680,std_ratio=3.8 ,pc_show=False):
    """
    Remove statistical outliers from the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
        nb_neighbors: int, The number of neighbors to use for radius/outlier removal.
        std_ratio: float, The standard deviation multiplier for the distance of points.
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_rm: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
    """
    if pc_data.shape[1] == 6:
        pcd_xyz = pc_data[:, :3]
        pcd_rgb = pc_data[:, 3:]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
        pcd_xyz_rm = rm.points
        pcd_rgb_rm = rm.colors
        pcd_rm = np.hstack((pcd_xyz_rm, pcd_rgb_rm))
        if pc_show:
            show_point_cloud(pcd_rm)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data)
        rm, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
        pcd_xyz_rm = rm.points
        pcd_rm = np.array(pcd_xyz_rm)
        if pc_show:
            show_point_cloud(pcd_rm)
    else:
        assert False, "Error: point cloud data shape is not correct!"

    return pcd_rm


def rm_radius_outlier(pc_data, nb_points=16, radius=0.05, pc_show=False):
    """
    Remove statistical outliers from the point cloud
    Parameter:
        pc_data: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
        nb_neighbors: int, The number of neighbors to use for radius/outlier removal.
        std_ratio: float, The standard deviation multiplier for the distance of points.
        pc_show: bool, whether to show the point cloud.
    Return:
        pcd_rm: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
    """
    if pc_data.shape[1] == 6:
        pcd_xyz = pc_data[:, :3]
        pcd_rgb = pc_data[:, 3:]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        rm, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        pcd_xyz_rm = rm.points
        pcd_rgb_rm = rm.colors
        pcd_rm = np.hstack((pcd_xyz_rm, pcd_rgb_rm))
        if pc_show:
            show_point_cloud(pcd_rm)
    elif pc_data.shape[1] == 3:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data)
        rm, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        pcd_xyz_rm = rm.points
        pcd_rm = np.array(pcd_xyz_rm)
        if pc_show:
            show_point_cloud(pcd_rm)
    else:
        assert False, "Error: point cloud data shape is not correct!"

    return pcd_rm



def add_GaussianNoise(noise_std_min, noise_std_max, pc_data):
    """
    Add Gaussian noise to the point cloud
    Parameter:
        noise_std_min: float, minimum of the Gaussian noise
        noise_std_max: float, maximum of the Gaussian noise
        pc_data: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
    Return:
        pc_noised: numpy array of pointcloud [[x, y, z, r, g, b], ...] or [[x, y, z], ...]
        noise_std: float, standard deviation of the Gaussian noise
    """
    pc_data = torch.tensor(pc_data)
    noise_std = random.uniform(noise_std_min, noise_std_max)
    pc_noised = pc_data + torch.randn_like(pc_data) * noise_std
    pc_noised = pc_noised.numpy()
    return pc_noised, noise_std


def rm_patches_pc(pc_data, patch_size, num_patches):
    """
    Args:
        pc_data: (N, 3) tensor
        patch_size: int, size of the patch
        num_patches: int, number of patches
    Returns:
        pc_incomplete: (M, 3) tensor
    """
    pc_data = torch.tensor(pc_data).float()  
    # get the id of the seed points
    N = pc_data.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pc_data[seed_idx].unsqueeze(0).float()     # (1, P, 3)
    
    # get the knn_idx of the remoing points of the seed patches
    _, knn_idx, _ = ops.knn_points(seed_pnts, pc_data.unsqueeze(0), K=patch_size, return_nn=False)
    
    # Remove points with knn_idx
    knn_idx = knn_idx.squeeze(0).view(-1)  # (P*K)
    mask = torch.ones(pc_data.shape[0], dtype=torch.bool)
    mask[knn_idx] = False
    pc_incomplete = pc_data[mask]
    pc_incomplete = pc_incomplete.numpy()
    
    return pc_incomplete

def random_spatial_crop_pc(pc_data, fraction=0.5, seed=None):
    """
    Randomly crop a spatial portion of the point cloud by removing
    approximately `fraction` of points in one random half-space.

    Steps:
      1) Compute centroid of the point cloud.
      2) Pick a random normal vector n in 3D.
      3) Project points (p - centroid) onto n.
      4) Determine a threshold so that approximately `fraction` of points
         lie on one side of the plane, then remove them.

    Args:
        pc_data: np.array of shape (N, 3) or (N, 6).
                 If (N, 6), columns 0..2 are assumed to be XYZ, and 3..5 are RGB.
        fraction: float in (0,1), fraction of points to remove.
                  e.g., fraction=0.5 => remove ~ half of the points
        seed: int or None, if set, random seed for reproducibility

    Returns:
        cropped_pc: np.array of shape (M, 3) or (M, 6), the remaining points.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Separate XYZ vs. RGB if present
    if pc_data.shape[1] == 3:
        xyz = pc_data
        rgb = None
    elif pc_data.shape[1] == 6:
        xyz = pc_data[:, :3]
        rgb = pc_data[:, 3:]
    else:
        raise ValueError("pc_data must be of shape (N,3) or (N,6).")

    # 1) Compute centroid
    centroid = np.mean(xyz, axis=0)  # shape (3,)

    # 2) Pick a random normal (n in 3D)
    #    We'll do a random direction with normal distribution, then normalize.
    n = np.random.randn(3)
    n = n / np.linalg.norm(n)  # shape (3,)

    # 3) Project each point onto n
    #    dot((p - centroid), n)
    projections = (xyz - centroid) @ n  # shape (N,)

    # 4) Determine threshold for fraction.
    #    Sort the projections, pick the quantile such that that fraction is removed.
    #    For fraction=0.5, this is effectively the median.
    threshold = np.quantile(projections, fraction)

    # 5) Randomly decide which side to remove (above or below the threshold)
    #    50% chance we remove points with projection > threshold
    #    50% chance we remove points with projection < threshold
    if random.random() < 0.5:
        # Remove points with projection > threshold
        mask = projections <= threshold
    else:
        # Remove points with projection < threshold
        mask = projections >= threshold

    xyz_cropped = xyz[mask]
    if rgb is not None:
        rgb_cropped = rgb[mask]
        cropped_pc = np.hstack([xyz_cropped, rgb_cropped])
    else:
        cropped_pc = xyz_cropped

    return cropped_pc

# ---------------------------------- Point cloud operations ----------------------------------------






if __name__ == '__main__':
    num_pts = 800
    mean = 0.88
    noise = np.random.normal(0, mean, (num_pts, 6))
    win8 = show_point_cloud(noise, point_size=0.01, bg_color=[1, 1, 1, 1], show_grid=False)
    win8.capture('./img/screenshot.png')
    # # import point cloud from the .pcd file
    # # path_to_point_cloud = './result/pcd/%06d.pcd' % int(2283)
    # # path_to_point_cloud = './result/pcd/%06d.pcd' % int(31)
    # # path_to_point_cloud = './result/pcd/single/%06d.pcd' % int(2283)
    # path_to_point_cloud = '/home/jared/SAIR_Lab/Super-Map/Grad-PU/data/PU-GAN/test_pointcloud/input_2048_4X/input_2048/pcd_down.xyz'
    # # point_cloud_data = read_pcd(path_to_point_cloud)  # little-endian float32
    # point_cloud_data = read_xyz(path_to_point_cloud)  # little-endian float32
    # window = show_point_cloud(point_cloud_data, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=True)

    # path_to_point_cloud2 = '/home/jared/SAIR_Lab/Super-Map/Grad-PU/pretrained_model/pugan/test/4X/pcd_down.xyz'
    # point_cloud_data2 = read_xyz(path_to_point_cloud2)  # little-endian float32
    # window2 = show_point_cloud(point_cloud_data2, point_size=0.01, bg_color=[0, 0, 0, 0], show_grid=True)

    # windows = [window, window2]
    # windows_close_ctrl(windows)

    # # downsample the point cloud to get the input
    # pcd_down = uniform_downsample_pc(point_cloud_data, pc_show=False)


    # # 1. Divide the point cloud into xyz and rgb.  2. Convert rgb from 0-255 to 0-1.
    # pcd_xyz, pcd_rgb, pcd_data = divide_pc_xyz_rgb(point_cloud_data, pc_show=True)


    # # add gaussian noise
    # pcd_xyz_out = noise_Gaussian(pcd_xyz, 0.3)
    # pcd_rgb_out = noise_Gaussian(pcd_rgb, 0.1)
    # pcd_out = np.hstack((pcd_xyz_out, pcd_rgb_out))

    # # Remove statistical outliers from the point cloud
    # pcd_rm = rm_statistical_outlier(pcd_out, nb_neighbors=680,std_ratio=3.8 ,pc_show=False)
     


# ---------------------------------- For pcd file (global map) ----------------------------------------
    # PATH = "./result/pcd/000031.pcd"
    # PATH = "/home/jared/Large_datasets/data/KITTI/KITTI_Tools/kitti-map-python/map-08_0.1_0-18.pcd"

    # # Simply view
    # pcd = o3d.io.read_point_cloud(PATH)
    # o3d.visualization.draw_geometries([pcd])
