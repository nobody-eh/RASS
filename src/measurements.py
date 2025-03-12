import os
import argparse
import json
import sys

import numpy as np
import pymeshlab as ml

from pose.convert_depth import compute_depth

def calc_real_camera_distance(scene_path):
    avg = 0
    c = 0
    with open(os.path.join(scene_path, "pose.txt"), "r") as f:
        for l in f:
            ln = float(l.split("\t")[-1].split("\n")[0])
            avg += 0.12 / ln
            c+=1
    return avg / c

def calc_real_camera_distance_using_depth_images(scene_path:str):
    sum = 0
    c = 0
    with open(os.path.join(scene_path, "pose.txt"), "r") as f:
        for l in f:
            id = l.split("\t")[0]
            sum += compute_depth(
                os.path.join(scene_path, 'depth', id),
                os.path.join(scene_path, 'masks_ref', id.replace('.jpg', '.png')),
                False, 5, (0, 3)
            )
            c+=1
    return sum / c


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='The dataset directory to scan')
    parser.add_argument('--diameter', type=float, default=5,
                        help='The diameter for removing isolated pieces in the mesh')

    args = parser.parse_args()

    scene_path = args.dataset_path
    #print(scene_path)
    gt_volumes = {"1" : 38.53,
                  "2" : 280.36,
                  "3" : 249.65,
                  "4" : 295.13,
                  "5" : 392.58,
                  "6" : 218.31,
                  "7" : 368.77,
                  "8" : 173.13,
                  "9" : 232.74,
                  "10": 163.23,
                  "11": 85.18,
                  "13": 308.28,
                  "14": 589.82}

    # Create a MeshSet object
    ms = ml.MeshSet()

    # Load the mesh (replace 'your_mesh.obj' with the path to your mesh file)
    ms.load_new_mesh(os.path.join(args.dataset_path, scene_path.split('/')[-1] + '.obj'))

    # clean up the mesh from small noises
    # Define the scaling factor along the x-axis
    diameter = ml.PercentageValue(args.diameter)

    # PyMeshLab 2023.12.post3 based on MeshLab 2023.12d
    # ml.print_pymeshlab_version()

    filters = ml.filter_list()
    # print(filters)

    # ml.print_filter_parameter_list('meshing_remove_connected_component_by_diameter')
    # Filters -> Scale, ... -> Transform: Scale, Normalise
    ms.apply_filter("meshing_remove_connected_component_by_diameter", mincomponentdiag=diameter, removeunref=True)

    # ml.print_filter_parameter_list('compute_matrix_from_scaling_or_normalization')
    # Get scale
    transforms_path = os.path.join(scene_path,'arcore2nerf_transforms.json')
    f = open(transforms_path)
    transforms = json.load(f)
    #actual_distance = calc_real_camera_distance(scene_path)
    #depth_distance = calc_real_camera_distance_using_depth_images(scene_path)
    normalized_scale = transforms['avglen']
    # print(normalized_scale, transforms['avglen'])

    # offset = transforms["offset"][0] ** 3
    # scale = normalized_scale + (normalized_scale * (transforms["scale"] * offset))
    scale = 1 / normalized_scale
    # scale = round(scale, 2)
    # print('Average Length=', scale)

    # Filters -> Scale, ... -> Transform: Scale, Normalise
    ms.apply_filter("compute_matrix_from_scaling_or_normalization", axisx=scale)

    # Apply orientation Filters -> Scale, ... -> Matrix: Set/Copy Transformation
    ms.apply_filter('set_matrix', transformmatrix=np.asarray(transforms['R']))
    # compute volumes
    measures = ms.apply_filter("get_geometric_measures")

    # todo: extract more values from here.
    # print("Mesh Volume: ", measures["mesh_volume"])

    # Estimate volume from clean watertight mesh
    volume = measures["mesh_volume"] * 10 ** 6
    # Apply scale as follows
    #volume_scaled = (10 ** 3) * (volume_ingp / (scale ** 3))
    scene_id = scene_path.split('/')[-1]
    gt = 0
    with open(os.path.join(scene_path, 'ground_truth.json'), "r") as f:
        f = json.load(f)
        gt = sum(item["food_volume"] for item in f)

    print(scene_id, round(volume,2), gt, normalized_scale)
    #print(f"{scene_id}\t{round(-volume, 2)}\t{gt_volumes[scene_id]}\t{round(gt_volumes[scene_id] - round(-volume, 2), 2)}\t{round(scale, 4)}\t{round(actual_distance,4)}\t{round(transforms['avglen']/100, 4)}\t{round(depth_distance, 4)}")
