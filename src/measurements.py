import os
import argparse
import json
import pymeshlab as ml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='The dataset directory to scan')
    parser.add_argument('--mesh', type=str,
                        help='The mesh path to scan (PLY/OBJ)')

    args = parser.parse_args()

    scene_path = args.dataset_path

    gt_volumes = {"1" : 38.53,
                  "2" : 280.36,
                  "3" : 249.65,
                  "4" : 295.13,
                  "5" : 392.58,
                  "6" : 218.31,
                  "7": 368.77,
                  "8" : 173.13,
                  "9" : 232.74,
                  "10": 163.23,
                  "11": 85.18,
                  "13": 308.28,
                  "14": 589.82}

    # Create a MeshSet object
    ms = ml.MeshSet()

    # Load the mesh (replace 'your_mesh.obj' with the path to your mesh file)
    ms.load_new_mesh(args.mesh)

    # ml.print_pymeshlab_version()
    filters = ml.filter_list()

    # ml.print_filter_parameter_list('compute_matrix_from_scaling_or_normalization')
    # Get scale
    transforms_path = os.path.join(scene_path,'transforms.json')
    f = open(transforms_path)
    transforms = json.load(f)
    normalized_scale = 12 / transforms['avglen'] * 4

    # print(normalized_scale, transforms['avglen'])

    offset = transforms["offset"][0] ** 3
    scale = normalized_scale + (normalized_scale * (transforms["scale"] + offset))
    # scale = round(scale, 2)
    # print('Average Length=', scale)

    # Filters -> Scale, ... -> Transform: Scale, Normalise
    ms.apply_filter("compute_matrix_from_scaling_or_normalization", axisx=scale, axisy=scale, axisz=scale)

    # compute volumes
    measures = ms.apply_filter("get_geometric_measures")

    # todo: extract more values from here.
    # print("Mesh Volume: ", measures["mesh_volume"])

    # Estimate volume from clean watertight mesh
    volume = measures["mesh_volume"]

    # Apply scale as follows
    #volume_scaled = (10 ** 3) * (volume_ingp / (scale ** 3))

    scene_id = scene_path.split('/')[-1]
    print(f"{scene_id}\t{round(-volume, 2)}\t{gt_volumes[scene_id]}\t{round(gt_volumes[scene_id] - round(-volume, 2), 2)}\t{scale}")