import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def decode_depth(rgb_depth, depth_range, filter_size) -> np.ndarray:
    if isinstance(rgb_depth, str):
        rgb_depth_data = cv2.imread(rgb_depth)
    else:
        rgb_depth_data = rgb_depth

    hsv_depth = cv2.cvtColor(rgb_depth_data, cv2.COLOR_BGR2HSV) / 255.
    depth_map = hsv_depth[:, :, 0] * 3  # Decode hue to meters (0-3m range)

    # Apply depth range filtering
    if depth_range:
        min_depth, max_depth = depth_range
        depth_map = np.clip(depth_map, min_depth, max_depth)

    # Apply noise reduction (optional)
    if filter_size > 0:
        depth_map = cv2.GaussianBlur(depth_map, (filter_size, filter_size), 0)

    return depth_map

def visualize_depth(depth_values: np.ndarray):
    """
    Visualizes depth values as a color map.

    Params:
    depth_values (np.ndarray): Depth values in meters.
    """
    plt.imshow(depth_values, cmap='viridis')
    plt.colorbar(label='Depth (meters)')
    plt.title('Depth Map Visualization')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.show()

def apply_mask(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a binary mask to the depth map.

    Params:
    depth_map (np.ndarray): Depth values in meters.
    mask (np.ndarray): Binary mask (grayscale or binary image).

    Returns:
    np.ndarray: Masked depth map.
    """
    # Ensure the mask is binary
    mask_binary = (mask > 0).astype(np.uint8)

    # Apply the mask to the depth map
    masked_depth = depth_map * mask_binary
    return masked_depth


def compute_depth(depth_path: str, mask_path: str, visualize: bool, filter_size: int, depth_range):
    """
    Main function to process the depth map and optionally visualize it.

    Params:
    depth_path (str): File path to the RGB depth image.
    mask_path (str): File path to the mask image (if applicable).
    visualize (bool): Flag to indicate whether to visualize the depth map.
    """
    # Read input images
    rgb_depth_data = cv2.imread(depth_path)
    rgb_mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if rgb_mask_data is None:
        print('Not found=', mask_path)
        return 0
    # Extract the depth map from the left half of the image
    depth_map = rgb_depth_data

    # Decode depth values
    depth_values = decode_depth(depth_map, depth_range=depth_range, filter_size=filter_size)

    # Apply the mask
    masked_depth_values = apply_mask(depth_values, rgb_mask_data)

    # Print average depth (excluding masked-out regions)
    valid_depths = masked_depth_values[masked_depth_values > 0]
    average_depth = np.average(valid_depths) if valid_depths.size > 0 else 0
    # print("Average Depth Value (meters):", average_depth)

    # Visualize depth map if specified
    if visualize:
        visualize_depth(masked_depth_values)

    return average_depth, valid_depths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode and process RGB depth maps.")
    parser.add_argument('--depth_path', type=str, required=True,
                        help='Path to the RGB depth image.')
    parser.add_argument('--depth_out', type=str, required=True,
                        help='Path to the output depth image.')
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to the mask image.')
    parser.add_argument('--depth_range', type=float, default=3.0, help="Maximum depth range in meters.")
    parser.add_argument('--filter_size', type=int, default=5, help="Kernel size for median filter.")
    parser.add_argument('--visualize_depth', action='store_true',
                        help='Flag to visualize the depth map.')

    args = parser.parse_args()
    # Explicitly pass parameters to main
    avg, msk = compute_depth(
        depth_path=args.depth_path,
        mask_path=args.mask_path,
        visualize=args.visualize_depth,
        depth_range=(0, args.depth_range),
        filter_size=args.filter_size
    )

    np.save(args.depth_out, msk)
    print(f"saved={args.depth_out}")
