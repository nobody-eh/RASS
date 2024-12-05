import os
import cv2
import numpy as np
from pathlib import Path
import copy
import argparse


if __name__ == '__main__':
    # Define argparse for handling command-line arguments
    parser = argparse.ArgumentParser(description='A small script to test do more experiments on ')
    parser.add_argument('--dataset_path', type=str, help='The directory path that contains masks')
    parser.add_argument('--chess_h', type=int, default=3, help='number of rows of the chessboard')
    parser.add_argument('--chess_w', type=int, default=4, help='number of columns of the chessboard')
    parser.add_argument('--vis', action='store_true', help='Visualize calibration?')

    args = parser.parse_args()
    data_root = parser.dataset_path
    chess_w = args.chess_w
    chess_h = args.chess_h
    for pth in os.listdir(os.path.join(data_root, 'poses')):
        with np.load(pth) as X:
            cmx, dist, _, _ = [X[i] for i in ('cmx', 'dist', 'rvecs', 'tvecs')]

            ret, corners = cv2.findChessboardCorners(binary, (chess_w, chess_h),
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                           cv2.CALIB_CB_FAST_CHECK +
                                                           cv2.CALIB_CB_NORMALIZE_IMAGE)

            object_points = np.zeros((chess_h * chess_w, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:chess_h, 0:chess_w].T.reshape(-1, 2)
            axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
            # compute transform
            #   - solvePnP requires camera calibraiton
            #   - the same info is also returned by calibrateCamera
            ret, rvec, tvec = cv2.solvePnP(object_points, corners, cmx, dist)

            print('scaling factor=', 0.012 / np.linalg.norm(tvec))