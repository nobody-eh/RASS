import copy
import json
import os
import time
import math

import cv2
import numpy as np


def calculate_camera_center(pose):
    try:
        pose = np.array(pose)
        R = pose[:3, :3]  # Extract the rotation matrix (3x3)
        t = pose[:3, 3]   # Extract the translation vector (3x1)
        camera_center = -np.dot(R.T, t)  # Calculate the camera center: -R^T * t
        return camera_center
    except Exception as e:
        print(f"Error processing pose: {e}")
        return None


def read_transform_save_ARCore_to_NERF_cams(cameras, sharpness_all, image_size, out_path):
    st_time = time.time()
    intrinsic, c2w_all, image_names, out = read_ARCoreCams(cameras, image_size)
    c2w_nerf, t_out = transform_ARCore_to_Nerf(c2w_all)
    save_nerf_cameras(c2w_nerf, sharpness_all, image_names, out, t_out, out_path)
    c2w_o3d = transform_Nerf_to_o3d(c2w_nerf)
    end_time = time.time() - st_time
    return intrinsic, c2w_o3d, end_time


def transform_ARCore_to_Nerf(cameras):

    ##################################################
    # Find scene center
    totp = find_scene_center_from_cams(cameras)
    ##################################################
    # Apply scene center and find up vector
    up = np.zeros(3)
    cameras_transform_1 = []
    for k, c2w in enumerate(cameras):
        c2w_last = copy.deepcopy(c2w)
        c2w_last[0:3, 3] = c2w_last[0:3, 3] - totp
        cameras_transform_1.append(c2w_last)
        up += c2w_last[0:3, 3]
    ##################################################
    # Rotate cameras to align with ingp format
    up_changed = up / np.linalg.norm(up)
    r = rotmat(up_changed, [1, 0, 0])  # rotate up vector to [1,0,0]
    r = np.pad(r, [0, 1])
    r[-1, -1] = 1

    cameras_transform_2 = []
    for k, c2w in enumerate(cameras_transform_1):
        c2w_last = copy.deepcopy(c2w)
        c2w_last = np.matmul(r, c2w_last)
        cameras_transform_2.append(c2w_last)
    ##################################################
    # Measure average distance from center and use it for  scene scaling
    # find a central point they are all looking at
    avglen = 0.
    cameras_transform_3 = []
    for c2w in cameras_transform_2:
        avglen += np.linalg.norm(c2w[0:3, 3])
    avglen /= len(cameras_transform_2)
    for k, c2w in enumerate(cameras_transform_2):
        c2w_last = copy.deepcopy(c2w)
        c2w_last[0:3, 3] *= (4.0 / avglen)
        cameras_transform_3.append(c2w_last)

    t_out = {
        "R": r.tolist(),
        "totp": totp.tolist(),
        "avglen": 4.0 / avglen}

    return cameras_transform_3, t_out


def find_scene_center_from_cams(cameras):
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for c2w_m in cameras:
        mf = c2w_m[0:3, :]
        for c2w_g in cameras:
            mg = c2w_g[0:3, :]
            p, w2 = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w2 > 0.00001:
                totp += p * w2
                totw += w2
    if totw > 0.0:
        totp /= totw
    return totp


def closest_point_2_lines(oa, da, ob, db):
    # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel

    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0

    return (oa + ta * da + ob + tb * db) * 0.5, denom


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def save_nerf_cameras(cameras, sharpness_all, image_names, out, t_out, out_path_scene):
    frames = []
    for img_name, img_sharpness, img_c2w in zip(image_names, sharpness_all, cameras):
        frame = {"file_path": "./images/" + img_name, "sharpness": img_sharpness,
                 'transform_matrix': copy.deepcopy(img_c2w).tolist()}
        frames.append(frame)

    out['frames'] = frames

    with open(os.path.join(out_path_scene, 'transforms.json'), "w") as outfile:
        json.dump(out, outfile, indent=2)

    with open(os.path.join(out_path_scene, 'arcore2nerf_transforms.json'), "w") as outfile:
        json.dump(t_out, outfile)


def transform_Nerf_to_o3d(cameras_matrix):
    upside_down_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    t_permutation_inv = np.array([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

    for k in range(len(cameras_matrix)):
        cameras_matrix[k] = np.matmul(t_permutation_inv, cameras_matrix[k])
        cameras_matrix[k] = np.matmul(cameras_matrix[k], upside_down_matrix)

    return cameras_matrix


def saveImagesFromVideo(scene_path):

    def sharpness(image):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm

    rgb_sequence_path = os.path.join(scene_path, 'images')
    os.mkdir(rgb_sequence_path)

    # SAVE VIDEO FRAMES
    vidcap = cv2.VideoCapture(os.path.join(scene_path, 'output_video.avi'))
    success, img = vidcap.read()
    h, w, _ = img.shape
    sharpness_all = []
    k = 0
    while success:
        sharpness_all.append(sharpness(img))
        cv2.imwrite(os.path.join(rgb_sequence_path, 'img_' + str(k).zfill(4) + '.jpg'), img)
        success, img = vidcap.read()
        k += 1

    return [sharpness_all, [w,h]]

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='The dataset directory to scan')

    args = parser.parse_args()

    scene_path = args.dataset_path

    apiinput = json.load(open(os.path.join(scene_path, 'apiinput.json')))

    cameras = {
        'fx': apiinput['cam_focal_length']['x'],
        'fy': apiinput['cam_focal_length']['y'],
        'cx': apiinput['cam_principal_point']['x'],
        'cy': apiinput['cam_principal_point']['y'],
        'model_mtx': apiinput['camera_poses'],
    }

    sharpness_all, image_size = saveImagesFromVideo(scene_path)
    intrinsic, c2w_o3d, read_cam_time = read_transform_save_ARCore_to_NERF_cams(cameras, sharpness_all, image_size, scene_path)