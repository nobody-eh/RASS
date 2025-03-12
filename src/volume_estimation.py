from colmap.read_write_model import read_points3D_binary
import argparse

import open3d as o3d
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='The dataset directory path to scan (PLY/OBJ)')

    args = parser.parse_args()

    point3d_path = os.path.join(args.dataset_path, 'sparse', 'geo', 'points3D.bin')

    points3D = read_points3D_binary(point3d_path)

    xyz = np.array([point.xyz for point in points3D.values()])
    rgb = np.array([point.rgb for point in points3D.values()])

    print(f"Loaded {xyz.shape[0]} points from COLMAP.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    o3d.visualization.draw_geometries([pcd])



