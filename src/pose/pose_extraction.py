import os
import cv2
import numpy as np
from pathlib import Path
import copy
import argparse

# for debugging
from plot import plotCamera
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseEstimation:
    def __init__(self, params):
        self.params = params
        self.data_root = self.params["data_root"]
        self.chess_T = self.params["chess_T"]
        self.chess_h = self.params["chess_h"]
        self.chess_w = self.params["chess_w"]
        self.vis = self.params["vis"]
        self.square_real_size = self.params["square_real_size"]
        # Prepare object points
        self.object_points = np.zeros((self.chess_h * self.chess_w, 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.chess_w, 0:self.chess_h].T.reshape(-1, 2)
        self.object_points *= self.square_real_size  # Scale to real-world square size

        self.focal_length_mm = self.params['focal_length_mm']  # Focal length in mm
        self.sensor_size_mm = self.params['sensor_size_mm'] # Sensor size in mm (diagonal for iPhone 12)

        self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        self.image_root = os.path.join(self.data_root, "images")

    def parseInt(self, imgpt):
        return (int(imgpt[0]), int(imgpt[1]))

    def draw(self, img, corners, imgpts):
        corner = self.parseInt(corners[0].ravel())

        cv2.line(img, corner, self.parseInt(imgpts[0].ravel()), (255, 0, 0), 5)
        cv2.line(img, corner, self.parseInt(imgpts[1].ravel()), (0, 255, 0), 5)
        cv2.line(img, corner, self.parseInt(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    def rtvec_to_matrix(self, rvec, tvec):
        """
        Convert rotation vector and translation vector to 4x4 matrix
        """
        rvec = np.asarray(rvec)
        tvec = np.asarray(tvec)

        T = np.eye(4)
        R, jac = cv2.Rodrigues(rvec)
        T[:3, :3] = R
        T[:3, 3] = tvec.squeeze()  # this is the fix
        return T

    def matrix_to_rtvec(self, matrix):
        """
        Convert 4x4 matrix to rotation vector and translation vector
        """
        rvec, jac = cv2.Rodrigues(matrix[:3, :3])
        tvec = matrix[:3, 3]
        return rvec, tvec

    def closest_point_2_lines(self, oa, da, ob, db):
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

    def find_scene_center_from_cams(self, cameras):
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for c2w_m in cameras:
            mf = c2w_m[0:3, :]
            for c2w_g in cameras:
                mg = c2w_g[0:3, :]
                p, w2 = self.closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w2 > 0.00001:
                    totp += p * w2
                    totw += w2
        if totw > 0.0:
            totp /= totw
        return totp

    def rotmat(self, a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        # handle exception for the opposite direction input
        if c < -1 + 1e-10:
            return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

    def transform_ARCore_to_Nerf(self, cameras):

        ##################################################
        # Find scene center
        totp = self.find_scene_center_from_cams(cameras)
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
        r = self.rotmat(up_changed, [1, 0, 0])  # rotate up vector to [1,0,0]
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

    def calculate_focal_length_in_pixels(self, focal_length_mm, image_resolution, sensor_size_mm):
        """
        Calculate focal length in pixels based on the given focal length in mm, image resolution, and sensor size.
        """
        # Assuming image resolution is (width, height)
        width_pixels, height_pixels = image_resolution
        # Calculate the focal length in pixels
        return focal_length_mm * (width_pixels / sensor_size_mm)

    def calculate_sensor_width(self, sensor_size_mm, resolution):
        """
        Calculate the sensor width given the diagonal sensor size and resolution.
        """
        width_pixels, height_pixels = resolution
        diagonal_pixels = np.sqrt(width_pixels**2 + height_pixels**2)
        aspect_ratio = width_pixels / diagonal_pixels
        sensor_width_mm = sensor_size_mm * aspect_ratio
        return sensor_width_mm

    def chessboard_detection(self, chessboard_pth, name_list, chess_T):
        corners_dict = {}
        useful_path = []
        useful_name = []
        # TODO: Cleanup this
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        bws = {}

        binary_dir = os.path.join(self.data_root, "binary")
        if not os.path.exists(binary_dir):
            os.makedirs(binary_dir)

        for pth, name in zip(chessboard_pth, name_list):
            cur_img = cv2.imread(pth)
            gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, chess_T, 255, cv2.THRESH_BINARY)[1]
            # Apply connected components labeling
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Extract the areas and component labels, excluding the background (label 0)
            component_areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]

            # Sort components by area (descending order) to find the largest ones
            sorted_components = sorted(component_areas, key=lambda x: x[1], reverse=True)

            # Select the two largest components
            largest_two_labels = [sorted_components[0][0], sorted_components[1][0], sorted_components[2][0],
                                  sorted_components[3][0], sorted_components[4][0]]

            # Create a mask to store convex hull regions
            convex_hull_mask = np.zeros_like(binary, dtype=np.uint8)  # Initialize a mask image

            # Morphological operation to fill holes (closing)
            kernel = np.ones((1, 1), np.uint8)  # You can adjust kernel size for larger holes
            for label in largest_two_labels:
                # Mask for the current component
                component_mask = np.uint8(labels == label) * 255

                # Fill holes using morphological close operation
                filled_component = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours of the filled component
                contours, _ = cv2.findContours(filled_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Apply Convex Hull to each contour
                for contour in contours:
                    hull = cv2.convexHull(contour)

                    # Draw the convex hull on the mask (as a white region)
                    cv2.drawContours(convex_hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

            # Apply the convex hull mask to the binary image using bitwise operations
            masked_image = cv2.bitwise_and(binary, convex_hull_mask)
            masked_image[convex_hull_mask == 0] = 255
            binary = masked_image
            # binary =  cv2.bitwise_not(binary)
            # save result
            # cv2.imshow('gray', cv2.resize(masked_image, (720, 960)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(pth.replace("images", "binary"), binary)
            chessboard_size = (self.chess_w, self.chess_h)
            ret, corners = cv2.findChessboardCorners(binary, chessboard_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                                    cv2.CALIB_CB_FAST_CHECK +
                                                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(binary, corners, (11, 11), (-1, -1), criteria)
                # Manually draw circles at the corners with custom thickness
                for corner in corners:
                    cv2.circle(cur_img, (int(corner[0][0]), int(corner[0][1])), radius=5, color=(255, 0, 0),
                               thickness=10)

                corners_dict[name] = corners
                useful_path += [pth]
                useful_name += [name]

                objpoints.append(self.object_points)
                imgpoints.append(corners)

                cv2.drawChessboardCorners(cur_img, chessboard_size, corners, ret)
                if self.vis:
                    cv2.imshow("fnl", cv2.resize(cur_img, (720, 960)))
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('s'):
                        cv2.imwrite(pth.replace("images", "chessboard"), cur_img)
                    cv2.destroyAllWindows()

                bws[pth.split('/')[-1]] = binary

        if len(useful_path) < (len(name_list) / 6):
            print(
                f"The number of chessboards that can be detected is less than {len(name_list) / 6}! readjusting the threshold "
                "chess_T !")
            print("Currently valid chessboard images: {}! Total {}!".format(len(useful_path), len(name_list)))
            if chess_T < 245:
                print(f"Retry with chess_T {chess_T + 5}")
                return self.chessboard_detection(chessboard_pth, name_list, chess_T + 5)
            else:
                print("Cannot find suitable chessboard")
                exit(0)
        else:
            print("Valid chessboard images detected: {}! Total {}!".format(len(useful_path), len(name_list)))

        # Find all related information
        for pth in bws:
            binary = bws[pth]
            # Obtain camera parameters
            sensor_width_mm = self.calculate_sensor_width(
                self.sensor_size_mm, binary.shape[::-1]
            )
            focal_length_pixels = self.calculate_focal_length_in_pixels(
                self.focal_length_mm, binary.shape[::-1], sensor_width_mm
            )

            image_size = binary.shape[::-1]
            center = (image_size[0] / 2, image_size[1] / 2)  # Principal point
            camera_matrix = np.array([
                [focal_length_pixels, 0, center[0]],
                [0, focal_length_pixels, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros(5)  # Assuming no distortion

            ret, cmx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, binary.shape[::-1], camera_matrix, dist_coeffs
            )

            # print(ret)
            # Re-projection Error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cmx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error

            print( "total error: {}".format(mean_error/len(objpoints)) )
            if ret > 2.0:
                print(f"root mean square error {ret} must be [0..1], skipping")
                continue
            # print(ret, cmx, dist, rvecs, tvecs)
            # save calibration result
            if not os.path.exists(os.path.join(self.data_root, 'poses')):
                os.makedirs(os.path.join(self.data_root, 'poses'))

            np.savez(os.path.join(self.data_root, 'poses', pth.split('.')[0] + '.npz'), cmx=cmx, dist=dist, rvecs=rvecs,
                     tvecs=tvecs)
        cam_locs = {}
        avg_len = {}
        for pth in bws:
            binary = bws[pth]
            with np.load(os.path.join(self.data_root, 'poses', pth.split('.')[0] + '.npz')) as X:
                cmx, dist, _, _ = [X[i] for i in ('cmx', 'dist', 'rvecs', 'tvecs')]

                ret, corners = cv2.findChessboardCorners(binary, (self.chess_w, self.chess_h),
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                               cv2.CALIB_CB_FAST_CHECK +
                                                               cv2.CALIB_CB_NORMALIZE_IMAGE)

                # compute transform
                #   - solvePnP requires camera calibraiton
                #   - the same info is also returned by calibrateCamera
                ret, rvec, tvec = cv2.solvePnP(self.object_points, corners, cmx, dist)
                avg_len[pth] = np.linalg.norm(tvec)

                # transform axis to images plane
                # axis_img, _ = cv2.projectPoints(self.axis, rvec, tvec, cmx, dist)

                # for obtaining the transformation matrix
                # print("Transformation Matrix:")
                # print(self.rtvec_to_matrix(rvec, tvec))

                Rt = cv2.Rodrigues(rvec)
                R = np.transpose(Rt[0])
                pos = -np.dot(R, tvec)
                cam_locs[pth] = pos.ravel()

        # Save for debugging
        with open(os.path.join(self.data_root, 'pose.txt'), 'w') as f:
            for pth in avg_len:
                f.write(f"{pth}\t{avg_len[pth]}\n")

        return cam_locs

    def load(self):
        name_list = os.listdir(self.image_root)[:50]
        name_list.sort()
        img_path_list = [os.path.join(Path(self.image_root), Path(name)) for name in name_list]

        cam_locs = self.chessboard_detection(img_path_list, name_list, self.chess_T)

        return cam_locs


if __name__ == "__main__":

    # Define argparse for handling command-line arguments
    parser = argparse.ArgumentParser(description='Camera caliberation using reference object for MTF dataset. This '
                                                 'script generates locations.txt file, where it used for '
                                                 'geo-registration of colmap')
    parser.add_argument('--dataset_path', type=str, help='The directory path that contains masks')
    parser.add_argument('--chess_h', type=int, default=3, help='number of rows of the chessboard')
    parser.add_argument('--chess_w', type=int, default=4, help='number of columns of the chessboard')
    parser.add_argument('--chess_T', type=int, default=190, help='binary thresholding of the chessboard')
    parser.add_argument('--square_real_size', type=float, default=0.012, help='Chessboard square size in millimeters')
    parser.add_argument('--focal_length_mm', type=float, default=26, help='Focal length of the camera in mm')
    parser.add_argument('--sensor_size_mm', type=float, default=4.8, help='Sensor size in mm (diagonal for iPhone 12)')
    parser.add_argument('--vis', action='store_true', help='Visualize calibration?')

    args = parser.parse_args()

    params = {
        "data_root": args.dataset_path,
        "chess_h": args.chess_h,
        "chess_w": args.chess_w,
        "chess_T": args.chess_T,
        "square_real_size": args.square_real_size,
        "focal_length_mm": args.focal_length_mm,
        "sensor_size_mm": args.sensor_size_mm,
        "vis": args.vis}
    est: PoseEstimation = PoseEstimation(params)

    cams_loc = est.load()
    loc_txt = os.path.join(os.path.join(est.data_root, 'locations.txt'))
    with open(loc_txt, 'w') as f:
        for pth in cams_loc:
            f.write(f"{pth}\t{cams_loc[pth][0]}\t{cams_loc[pth][1]}\t{cams_loc[pth][2]}\n")

    print(f"written {len(cams_loc)} to {loc_txt}")
