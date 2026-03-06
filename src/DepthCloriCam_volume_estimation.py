import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mask_path', type=str,
                    help='The mask image path')
parser.add_argument('--depth_path', type=str,
                    help='depth data')
parser.add_argument('--use_morphology', type=bool,
                    help='enable morphology on depth map to minimize noises using 5x5 kernel',
                    default=True)
parser.add_argument('--vis_morphologied_depth', type=bool,
                    help='enable morphology on depth map to minimize noises',
                    default=False)

args = parser.parse_args()

imgs_path = args.mask_path
depth_path = args.depth_path

##### Convert the mask into bw
# Read a grayscale image
im_gray = cv2.imread(imgs_path, cv2.IMREAD_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY)

# cv2.imshow("Black and White", im_bw)
# cv2.waitKey(0)

# get the depth information to measure the third coordinate
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

d = depth
if args.use_morphology:
    # We need to mask the depth information
    d = np.zeros(depth.shape, np.uint16)
    for x, y in np.argwhere(im_bw == 255):
        d[x, y] = depth[x, y]

    kernel_size = (5, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, kernel)

    if args.vis_morphologied_depth:
        d2 = np.zeros(depth.shape)
        for x, y in np.argwhere(d > 0):
            d2[x, y] = 255

        # applying morphology closing
        cv2.imshow("close", d2)
        cv2.waitKey(0)

volume = 0
# the distance between camera and capture plane
# applied equation 6, 7
# https://dl-acm-org.sire.ub.edu/doi/pdf/10.1145/3347448.3357172
# See section 4.4 https://arxiv.org/pdf/2103.03375.pdf
z_ref = 35.9
# 5.957 × 10^−3 cm^2
s_pixel = 0.005957
for x, y in np.argwhere(im_bw == 255):
    # We found some values in the depth map are bigger than the z_ref. In this case, we need to point into z_ref to avoid
    # negative vales.
    depth_cm = d[x, y] / 100

    z_p = depth_cm if d[x, y] > 0 and depth_cm < z_ref else z_ref

    r_p = pow(z_ref / z_p, 2)

    volume += (s_pixel / r_p) * (z_ref - z_p)

# count the number of pixels
print(volume)