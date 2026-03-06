import argparse
import json
import os
import numpy as np
import cv2
import math

try:
    from skimage import io as skio
    from skimage.color import rgb2gray as sk_rgb2gray
    from skimage.measure import shannon_entropy as sk_shannon_entropy

    HAS_SKIMAGE = True
except Exception:
    skio = None
    sk_rgb2gray = None
    sk_shannon_entropy = None
    HAS_SKIMAGE = False

# Number of keypoints/features per image: How many 2D features were detected; richness of feature content per view.
# Number of observed 3D points per image: How many of those keypoints are successfully triangulated; view completeness.
# Observation ratio: (observed 3D points) / (total keypoints): Measure of how well each view contributes to 3D reconstruction.
# Camera pose / camera center: Compute the camera center from the pose (quaternion + translation). Used for viewpoint diversity, positional spread.
# Distances of camera centers from scene origin or mean center: How spread out the views are spatially.
# Orientation / view direction: Derive the view direction of each image from pose; compute variation / angular spread.
# Image name / camera ID: You can use camera ID or intrinsics grouping to determine whether some images use different lenses or camera settings.
# Number of images in the scene (Count of image entries in images.txt) ‒ gives scene coverage.
def parse_images_txt(images_txt_path):
    """
    Parses COLMAP images.txt.
    Returns:
        images_info: list of dicts, each:
            {
              'image_id': int,
              'quat': (qw, qx, qy, qz),
              't': (tx, ty, tz),
              'camera_id': int,
              'image_name': str,
              'points2d': list of (x, y, point3D_id)
            }
    """
    images = []
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        # Next line: POINTS2D
        i += 1
        if i >= len(lines):
            break
        pts_line = lines[i].strip()
        pts2d = []
        if pts_line:
            pparts = pts_line.split()
            # Each feature: 3 entries (x, y, point3D_id)
            for j in range(0, len(pparts), 3):
                try:
                    x = float(pparts[j])
                    y = float(pparts[j+1])
                    pid = int(pparts[j+2])
                except:
                    continue
                pts2d.append((x, y, pid))
        images.append({
            'image_id': image_id,
            'quat': (qw, qx, qy, qz),
            't': (tx, ty, tz),
            'camera_id': camera_id,
            'image_name': image_name,
            'points2d': pts2d
        })
        i += 1
    return images

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts quaternion (Hamiltonian convention) to 3x3 rotation matrix.
    """
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    w, x, y, z = qw, qx, qy, qz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),       2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)]
    ])
    return R

def compute_images_txt_features(images, image_width=None, image_height=None):
    """
    Given parsed images list, compute features per-image then aggregate per scene.
    Returns a dict of aggregated features.
    """
    visible_counts = []
    visible_ratios = []
    cam_centers = []
    view_dirs = []
    total_images = len(images)

    for img in images:
        pts2d = img['points2d']
        total_kpts = len(pts2d)
        if total_kpts == 0:
            visible_counts.append(0)
            visible_ratios.append(0.0)
        else:
            vis3d = sum(1 for (_x, _y, pid) in pts2d if pid >= 0)
            visible_counts.append(vis3d)
            visible_ratios.append(vis3d / total_kpts)

        # Compute camera center from (quat, t)
        qw, qx, qy, qz = img['quat']
        tx, ty, tz = img['t']
        R = quat_to_rotation_matrix(qw, qx, qy, qz)
        T = np.array([tx, ty, tz], dtype=float)
        C = - R.T.dot(T)
        cam_centers.append(C)

        # View direction: assume camera looks along its local +Z axis
        # So view_dir = R.T * [0,0,1]
        vd = R.T.dot(np.array([0.0, 0.0, 1.0], dtype=float))
        vd = vd / (np.linalg.norm(vd) + 1e-12)
        view_dirs.append(vd)

    # convert to numpy
    visible_counts = np.array(visible_counts, dtype=float)
    visible_ratios = np.array(visible_ratios, dtype=float)
    cam_centers = np.array(cam_centers, dtype=float)
    view_dirs = np.array(view_dirs, dtype=float)

    # Camera center distances
    dists = np.linalg.norm(cam_centers, axis=1) if cam_centers.shape[0] > 0 else np.array([])

    # Orientation variation: difference angles between view_dirs and mean view_dir
    if view_dirs.shape[0] > 0:
        mean_vd = view_dirs.mean(axis=0)
        mean_vd = mean_vd / (np.linalg.norm(mean_vd) + 1e-12)
        dot = view_dirs.dot(mean_vd)
        dot = np.clip(dot, -1.0, 1.0)
        angles = np.arccos(dot)
    else:
        angles = np.array([])

    # Now aggregate
    features = {
        'num_images': total_images,
        'mean_visible_points': float(np.mean(visible_counts)) if total_images > 0 else float('nan'),
        'std_visible_points': float(np.std(visible_counts, ddof=0)) if total_images > 0 else float('nan'),
        'mean_visible_ratio': float(np.mean(visible_ratios)) if total_images > 0 else float('nan'),
        'std_visible_ratio': float(np.std(visible_ratios, ddof=0)) if total_images > 0 else float('nan'),
        'mean_camera_center_dist': float(np.mean(dists)) if dists.size > 0 else float('nan'),
        'std_camera_center_dist': float(np.std(dists, ddof=0)) if dists.size > 0 else float('nan'),
        'mean_view_angle_var': float(np.mean(angles)) if angles.size > 0 else float('nan'),
        'std_view_angle_var': float(np.std(angles, ddof=0)) if angles.size > 0 else float('nan'),
    }
    return features

# PointCloud features
def parse_colmap_points3d_txt(txt_path):
    """
    Parse points3D.txt in COLMAP text format.

    Returns:
       list of dicts, each with keys: xyz (tuple of 3 floats), error (float), track_length (int)
    """
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # At least up to ERROR and one track observation
            # Format: ID X Y Z R G B ERROR TRACK…
            # So parts[0] = point id
            # parts[1:4] = X, Y, Z
            # parts[4:7] = R, G, B
            # parts[7] = ERROR
            # parts[8:] = TRACK entries (2 × track_length entries)
            if len(parts) < 8:
                continue
            # parse
            x, y, z = map(float, parts[1:4])
            error = float(parts[7])
            # track entries are image_id, point2d_idx pairs
            track_entries = parts[8:]
            track_length = len(track_entries) // 2  # each track has 2 tokens
            points.append({
                'xyz': (x, y, z),
                'error': error,
                'track_length': track_length
            })
    return points

def compute_point_cloud_features_from_txt(txt_path):
    """
    Given path to points3D.txt, compute:
        - point count
        - bounding box (min, max)
        - extents
        - volume (axis-aligned box)
        - point density = point_count / volume
        - error: mean,std,median
        - track_length: mean,std,median
    Returns a dict of features.
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"points3D.txt not found at {txt_path}")
    pts = parse_colmap_points3d_txt(txt_path)
    if len(pts) == 0:
        # no points
        return {
            'point_count': 0,
            'volume': float('nan'),
            'point_density': float('nan'),
            'mean_error': float('nan'),
            'std_error': float('nan'),
            'median_error': float('nan'),
            'mean_track_length': float('nan'),
            'std_track_length': float('nan'),
            'median_track_length': float('nan')
        }
    coords = np.array([p['xyz'] for p in pts], dtype=float)
    errors = np.array([p['error'] for p in pts], dtype=float)
    tracks = np.array([p['track_length'] for p in pts], dtype=int)

    # bbox
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    extents = bbox_max - bbox_min
    # Avoid zero extents
    eps = 1e-12
    volume = float(max(extents[0], eps) * max(extents[1], eps) * max(extents[2], eps))

    point_count = coords.shape[0]
    point_density = point_count / volume

    feat = {
        'point_count': int(point_count),
        'bbox_min_x': float(bbox_min[0]),
        'bbox_min_y': float(bbox_min[1]),
        'bbox_min_z': float(bbox_min[2]),
        'bbox_max_x': float(bbox_max[0]),
        'bbox_max_y': float(bbox_max[1]),
        'bbox_max_z': float(bbox_max[2]),
        'extent_x': float(extents[0]),
        'extent_y': float(extents[1]),
        'extent_z': float(extents[2]),
        'volume': float(volume),
        'point_density': float(point_density),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors, ddof=0)),
        'median_error': float(np.median(errors)),
        'mean_track_length': float(np.mean(tracks)),
        'std_track_length': float(np.std(tracks, ddof=0)),
        'median_track_length': float(np.median(tracks))
    }
    return feat


# Sobel edge fraction, texture/edge density (Sobel edge-fraction) to our script
def sobel_edge_fraction(img_gray, threshold=None):
    """
    Compute the fraction of pixels that are edges according to Sobel gradient magnitude.

    Args:
        img_gray: 2D numpy array, grayscale image (uint8 or float scaled).
        threshold: float or None. If None, use mean gradient magnitude or a percentile as threshold.

    Returns:
        edge_fraction: float in [0,1]
    """
    # Compute Sobel in x and y directions
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    grad_mag = np.sqrt(gx * gx + gy * gy)

    # Normalize optionally or leave as is
    # Could use grad_mag.mean(), or a percentile (e.g. 75th) to set threshold
    if threshold is None:
        threshold = np.mean(grad_mag)
    # Or threshold = np.percentile(grad_mag, 75)

    # Compute fraction of pixels above threshold
    mask = grad_mag > threshold
    edge_frac = mask.sum() / mask.size
    return edge_frac

def compute_scene_edge_density(image_paths, convert_to_gray=True, threshold_strategy='mean'):
    """
    Given a list of image paths for a scene, compute avg & std of edge fractions.

    Args:
       image_paths: list of str
       convert_to_gray: whether to convert color images to grayscale
       threshold_strategy: 'mean' or 'percentile' or fixed value

    Returns:
       (mean_edge_frac, std_edge_frac)
    """
    edge_fracs = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if convert_to_gray and img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        if threshold_strategy == 'mean':
            thresh = None
        elif threshold_strategy.startswith('percentile'):
            # e.g. 'percentile_75'
            _, perc = threshold_strategy.split('_')
            perc = float(perc)
            # compute grad first
            gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx * gx + gy * gy)
            thresh = np.percentile(grad_mag, perc)
        else:
            # if threshold_strategy is numeric
            try:
                thresh = float(threshold_strategy)
            except ValueError:
                thresh = None

        edge_frac = sobel_edge_fraction(img_gray, threshold=thresh)
        edge_fracs.append(edge_frac)

    if not edge_fracs:
        return (None, None)
    return (float(np.mean(edge_fracs)), float(np.std(edge_fracs)))


# Compute the masks statistics
def compute_foreground_stats_for_image(mask):
    """
    mask: binary mask (0 background, 1 foreground) or grayscale/float mask thresholded
    Returns: dict with area_fraction, num_components, largest_component_fraction, bbox_aspect_ratio, etc.
    """
    h, w = mask.shape
    total_pixels = h * w

    # Ensure binary
    bin_mask = (mask > 0).astype(np.uint8)

    # Area fraction
    area = bin_mask.sum()
    area_frac = area / total_pixels

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # Note: stats has shape (num_labels, 5): [x, y, width, height, area]
    # The label 0 is background (so skip stats[0])
    if num_labels <= 1:
        # No foreground
        largest_comp_frac = 0.0
        bbox_aspect_ratio = 0.0
    else:
        # foreground components are labels 1..num_labels-1
        comps = stats[1:]  # skip label 0
        areas = comps[:, cv2.CC_STAT_AREA]
        widths = comps[:, cv2.CC_STAT_WIDTH]
        heights = comps[:, cv2.CC_STAT_HEIGHT]

        largest_area = areas.max()
        largest_comp_frac = largest_area / total_pixels

        # For the largest component, compute aspect ratio
        idx = areas.argmax()
        w_comp = widths[idx]
        h_comp = heights[idx]
        if h_comp > 0:
            bbox_aspect_ratio = w_comp / h_comp
        else:
            bbox_aspect_ratio = 0.0

    return {
        'area_frac': area_frac,
        'num_fg_components': num_labels - 1,  # subtract background
        'largest_comp_frac': largest_comp_frac,
        'bbox_aspect_ratio': bbox_aspect_ratio
    }

def extract_texture_density_scene_features(image_paths):
    """
    Example feature extractor for one scene directory.
    scene_dir: path to a scene's image folder
    Assumes images are under scene_dir/images/*.png (or adjust accordingly)
    """
    # (Other feature computations here...)
    # Now compute edge density
    mean_edge, std_edge = compute_scene_edge_density(image_paths, threshold_strategy='mean')
    return {
        'edge_frac_mean': mean_edge,
        'edge_frac_std': std_edge,
        # ... other features ...
    }

def compute_scene_foreground_stats(mask_paths):
    """
    mask_paths: list of file paths to masks for all images in a scene
    Returns aggregated stats: mean & std of area_frac etc.
    """
    stats_list = []
    for mp in mask_paths:
        if not os.path.isfile(mp):
            continue
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        stats = compute_foreground_stats_for_image(mask)
        stats_list.append(stats)

    agg = {}
    if not stats_list:
        # scene has no masks or none loaded
        agg['area_frac_mean'] = float('nan')
        agg['area_frac_std'] = float('nan')
        agg['num_components_mean'] = float('nan')
        agg['num_components_std'] = float('nan')
        agg['largest_comp_frac_mean'] = float('nan')
        agg['largest_comp_frac_std'] = float('nan')
        agg['bbox_aspect_ratio_mean'] = float('nan')
        agg['bbox_aspect_ratio_std'] = float('nan')
        return agg

    # Aggregate
    agg = {}
    # collect arrays
    area_fracs = np.array([s['area_frac'] for s in stats_list])
    num_comps = np.array([s['num_fg_components'] for s in stats_list])
    largest_fracs = np.array([s['largest_comp_frac'] for s in stats_list])
    aspect_ratios = np.array([s['bbox_aspect_ratio'] for s in stats_list])

    agg['area_frac_mean'] = float(area_fracs.mean())
    agg['area_frac_std'] = float(area_fracs.std())
    agg['num_components_mean'] = float(num_comps.mean())
    agg['num_components_std'] = float(num_comps.std())
    agg['largest_comp_frac_mean'] = float(largest_fracs.mean())
    agg['largest_comp_frac_std'] = float(largest_fracs.std())
    agg['bbox_aspect_ratio_mean'] = float(aspect_ratios.mean())
    agg['bbox_aspect_ratio_std'] = float(aspect_ratios.std())

    return agg


# Compute image entropy
def _to_uint8(img):
    arr = np.asarray(img)
    if arr.size == 0:
        return arr.astype(np.uint8)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if 0.0 <= mn and mx <= 1.0:
            arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _read_gray_uint8(path):
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _to_uint8(img)


def _entropy_from_gray_uint8(gray):
    if gray is None:
        return None
    if HAS_SKIMAGE:
        try:
            return float(sk_shannon_entropy(gray))
        except Exception:
            pass
    return float(manual_entropy(gray))


def image_entropy(path, use_grayscale=True):
    if not os.path.isfile(path):
        return None

    # Prefer skimage when available.
    if HAS_SKIMAGE:
        try:
            img = skio.imread(path)
        except Exception:
            img = None
        if img is not None:
            if use_grayscale:
                if img.ndim >= 3:
                    if img.ndim == 4:
                        img = img[0]
                    try:
                        img_gray = sk_rgb2gray(img)
                    except Exception:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img
                img_gray = _to_uint8(img_gray)
                return float(sk_shannon_entropy(img_gray))

            if img.ndim < 3:
                return float(sk_shannon_entropy(_to_uint8(img)))
            ent = [float(sk_shannon_entropy(_to_uint8(img[..., c]))) for c in range(img.shape[-1])]
            return float(np.mean(ent))

    # Fallback path without skimage.
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if use_grayscale:
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(manual_entropy(_to_uint8(img)))

    if img.ndim < 3:
        return float(manual_entropy(_to_uint8(img)))
    ent = [manual_entropy(_to_uint8(img[..., c])) for c in range(img.shape[-1])]
    return float(np.mean(ent))

def manual_entropy(img_gray, bins=256):
    # img_gray assumed uint8 grayscale
    if img_gray is None:
        return float('nan')
    if img_gray.size == 0:
        return float('nan')
    hist, _ = np.histogram(img_gray, bins=bins, range=(0,256))
    total = hist.sum()
    if total <= 0:
        return float('nan')
    probs = hist / total
    probs = probs[probs > 0]  # remove zero entries
    H = -np.sum(probs * np.log2(probs))
    return float(H)


def fov_to_focal(fov_rad, res):
    return 0.5 * res / np.tan(0.5 * fov_rad)


def _expand_mask_dirs(mask_dirs):
    expanded = []
    for md in mask_dirs:
        m = str(md).strip("/").replace("\\", "/")
        if not m:
            continue
        expanded.append(m)
        if m == "masks":
            # Common Nutrition5k layout.
            expanded.append("masks/Annotations")
            expanded.append("masks/Visualizations")
    # stable de-dup
    seen = set()
    out = []
    for x in expanded:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _mask_relpath_candidates(frame_path, mask_dirs):
    norm = str(frame_path).replace("\\", "/").lstrip("./")
    candidates = []
    roots = _expand_mask_dirs(mask_dirs)

    if norm.startswith("images/"):
        suffix = norm[len("images/") :]
        for md in roots:
            candidates.append(f"{md}/{suffix}")

    base = os.path.basename(norm)
    for md in roots:
        candidates.append(f"{md}/{base}")

    # stable de-dup
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_mask_path(dataset_path, frame_path, mask_dirs=("masks_omvs", "masks", "rgba")):
    for rel in _mask_relpath_candidates(frame_path, mask_dirs):
        p = os.path.join(dataset_path, rel)
        if os.path.isfile(p):
            return p
    return None


def _select_frames(frames, max_frames=None, frame_stride=1):
    stride = max(1, int(frame_stride))
    sampled = list(frames[::stride])
    if max_frames is None:
        return sampled
    k = max(1, int(max_frames))
    if len(sampled) <= k:
        return sampled
    # Deterministic uniform downsampling across selected frames.
    idx = np.linspace(0, len(sampled) - 1, num=k, dtype=int)
    return [sampled[i] for i in idx]


def extract_features(dataset_path, data, max_frames=None, frame_stride=1):
    meta = {}
    w = data.get('w')
    h = data.get('h')

    if 'fl_x' in data:
        meta['fl_x'] = data['fl_x']
    elif data.get('camera_angle_x') and w:
        meta['fl_x'] = fov_to_focal(data['camera_angle_x'], w)
    else:
        meta['fl_x'] = None

    if 'fl_y' in data:
        meta['fl_y'] = data['fl_y']
    elif data.get('camera_angle_y') and h:
        meta['fl_y'] = fov_to_focal(data['camera_angle_y'], h)
    else:
        meta['fl_y'] = meta['fl_x']

    meta['camera_angle_x'] = data.get('camera_angle_x')
    meta['camera_angle_y'] = data.get('camera_angle_y')

    for key in ['k1', 'k2', 'k3', 'p1', 'p2', 'cx', 'cy']:
        if key in data:
            meta[key] = data[key]

    meta['w'] = w
    meta['h'] = h
    if 'aabb_scale' in data:
        meta['aabb_scale'] = data['aabb_scale']

    all_frames = data.get('frames', [])
    frames = _select_frames(all_frames, max_frames=max_frames, frame_stride=frame_stride)
    meta['num_frames_total'] = len(all_frames)
    meta['num_frames_used'] = len(frames)

    sharp = []
    entropy = []
    edge_fracs = []
    masks_paths = []
    for frame in frames:
        sharp.append(frame.get('sharpness'))
        frame_path = frame['file_path']
        img_path = os.path.join(dataset_path, frame_path)
        gray = _read_gray_uint8(img_path)
        e = _entropy_from_gray_uint8(gray)
        if e is not None:
            entropy.append(e)
        if gray is not None:
            edge_fracs.append(float(sobel_edge_fraction(gray, threshold=None)))
        mask_path = resolve_mask_path(dataset_path, frame_path)
        if mask_path is not None:
            masks_paths.append(mask_path)


    sharp_vals = [float(v) for v in sharp if v is not None]
    if sharp_vals:
        meta['sharpness_mean'] = float(np.mean(sharp_vals))
        meta['sharpness_std'] = float(np.std(sharp_vals))
    else:
        meta['sharpness_mean'] = None
        meta['sharpness_std'] = None

    if entropy:
        meta["entropy_mean"] = float(np.mean(entropy))
        meta["entropy_std"] = float(np.std(entropy))
    else:
        meta["entropy_mean"] = None
        meta["entropy_std"] = None

    if edge_fracs:
        meta["edge_frac_mean"] = float(np.mean(edge_fracs))
        meta["edge_frac_std"] = float(np.std(edge_fracs))
    else:
        meta["edge_frac_mean"] = None
        meta["edge_frac_std"] = None

    meta.update(
        compute_scene_foreground_stats(masks_paths)
    )

    meta.update(
        compute_point_cloud_features_from_txt(os.path.join(dataset_path, 'sparse', 'txt', 'points3D.txt'))
    )

    images_txt_path = os.path.join(dataset_path, 'sparse', 'txt', 'images.txt')
    imgs = parse_images_txt(images_txt_path)
    feats = compute_images_txt_features(imgs)
    meta.update(feats)

    return meta

def main():
    parser = argparse.ArgumentParser(description="Extract camera scene features and output one CSV line")
    parser.add_argument("input", help="Path to transforms.json")
    args = parser.parse_args()
    dataset_path = os.path.dirname(args.input)
    with open(args.input, 'r') as f:
        data = json.load(f)

    features = extract_features(dataset_path, data)
    output = ', '.join(f"{k}={features[k]}" for k in sorted(features.keys()))
    print(output)

if __name__ == "__main__":
    main()
