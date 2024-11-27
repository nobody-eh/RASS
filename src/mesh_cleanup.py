import pymeshlab as ml

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str,
                        help='The mesh path to scan (PLY/OBJ)')

    parser.add_argument('--mesh_cleaned_path', type=str,
                        help='The cleaned mesh path to save (PLY/OBJ)')

    parser.add_argument('--diameter', type=float, default=5,
                        help='The diameter for removing isolated pieces in the mesh')
    args = parser.parse_args()

    mesh = args.mesh_path
    mesh_cleaned = args.mesh_cleaned_path

    # if os.path.exists(mesh_cleaned):
    #     print("Skip=", mesh_cleaned)
    #     exit(0)
    # Create a MeshSet object
    ms = ml.MeshSet()

    # Load the mesh (replace 'your_mesh.obj' with the path to your mesh file)
    ms.load_new_mesh(mesh)

    # Define the scaling factor along the x-axis
    diameter = ml.Percentage(args.diameter)

    ml.print_pymeshlab_version()
    filters = ml.filter_list()
    # print(filters)

    ml.print_filter_parameter_list('meshing_remove_connected_component_by_diameter')
    # Filters -> Scale, ... -> Transform: Scale, Normalise
    ms.apply_filter("meshing_remove_connected_component_by_diameter", mincomponentdiag=diameter, removeunref=True)

    # compute volumes
    ms.save_current_mesh(mesh_cleaned)

    print("Saved=", mesh_cleaned)
