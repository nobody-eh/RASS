import argparse
import json

import numpy as np
import open3d as o3d
import os

def compute_mesh_volume(mesh):
    """Computes the volume of a watertight mesh using Open3D."""
    mesh.compute_vertex_normals()
    if not mesh.is_watertight():
        raise ValueError("Mesh is not watertight. Volume computation may be incorrect.")

    volume = mesh.get_volume()
    return volume


def main():
    parser = argparse.ArgumentParser(description="Compute the volume of a 3D mesh.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--mesh_out", type=str, required=True, help="Path to the mesh output file.")
    # parser.add_argument("--scale", type=float, required=True, help="Scaling factor for the mesh.")
    parser.add_argument("--transform", type=str, required=True, help="Path to the transformation matrix (numpy file).")
    args = parser.parse_args()

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)

    if not mesh:
        raise ValueError("Failed to load mesh. Check the file path.")

    with open(os.path.join(args.transform)) as f:
        transform = json.load(f)
        transform_matrix = np.asarray(transform["R"])
        scale = transform["avglen"]
    # Apply scaling
    mesh.scale(1 / scale, center=(0, 0, 0))

    # # Load transformation matrix
    # with np.load(os.path.join(args.transform)) as X:
    #     cmx, dist, _, _ = [X[i] for i in ('cmx', 'dist', 'rvecs', 'tvecs')]
    #     print(cmx)

    # transform_matrix = np.load(args.transform)

    # Transform to origin coordinate system
    mesh.transform(np.linalg.inv(transform_matrix))

    o3d.io.write_triangle_mesh(args.mesh_out, mesh)

    print(args.mesh_out)

    # Compute volume
    # volume = compute_mesh_volume(mesh)
    # print(f"Computed Volume: {volume:.6f} cubic units")

if __name__ == "__main__":
    main()
