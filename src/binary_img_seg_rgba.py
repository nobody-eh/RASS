from PIL import Image
import os
import numpy as np
import shutil

def create_rgba_images(mask_dir, rgb_dir, output_dir):
    mask_files = os.listdir(mask_dir)
    rgb_files = os.listdir(rgb_dir)

    for mask_file in mask_files:
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            rgb_file = mask_file.replace('.png', '.jpg')  # Adjust the renaming to match RGB file
            rgb_path = os.path.join(rgb_dir, rgb_file)
            if rgb_file in rgb_files:
                mask = np.array(Image.open(mask_path).convert("L"))
                rgb = np.array(Image.open(rgb_path))

                # Ensure the mask and RGB images have the same dimensions
                if mask.shape != rgb.shape[:2]:
                    print(f"Skipping: Mask {mask_file} and RGB {rgb_file} images must have the same dimensions.")
                    continue

                # Create a new RGBA image with transparency based on the mask
                rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)

                # Preserve RGB pixels where the mask is not black
                rgba[..., :3] = rgb

                # Set alpha channel to 255 for non-black pixels in the mask
                rgba[..., 3] = np.where(mask == 0, 0, 255)

                # Save the RGBA image with the same name as the original RGB image
                rgba_image = Image.fromarray(rgba)
                rgba_path = os.path.join(output_dir, rgb_file.replace('.jpg', '.png'))
                rgba_image.save(rgba_path)
                print(f"Saved python RGBA image: {rgba_path}")
            else:
                print(f"No corresponding RGB image found for mask: {mask_file}")


if __name__ == '__main__':
    import argparse  # Added import for argparse

    # Define argparse for handling command-line arguments
    parser = argparse.ArgumentParser(description='Binary Image Segmentation: Accept RGB Image and Mask to generate a '
                                                 'transparent Image')
    parser.add_argument('--mask_dir', type=str, help='The directory path that contains masks')
    parser.add_argument('--rgb_dir', type=str, help='The directory path that contains images')
    parser.add_argument('--output_dir', type=str, help='The directory path that contains images')

    args = parser.parse_args()

    mask_dir = args.mask_dir
    rgb_dir = args.rgb_dir
    output_dir = args.output_dir
    import os
    # clean up
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    create_rgba_images(mask_dir, rgb_dir, output_dir)
