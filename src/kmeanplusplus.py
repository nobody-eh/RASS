import argparse
import cv2
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, disk
import faiss
import matplotlib.pyplot as plt
import os


def preprocess_image(img):
    # Apply Gaussian filter
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert RGB to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # First thresholding using Otsu's method
    _, thresh1 = cv2.threshold(hsv_img[:, :, 1], args.threshold1, 255, cv2.THRESH_BINARY)

    # Second thresholding using Otsu's method
    _, thresh2 = cv2.threshold(thresh1, args.threshold2, 255, cv2.THRESH_BINARY)

    # Combine thresholds
    final_thresh = cv2.bitwise_and(img, img, mask=thresh2)

    return cv2.cvtColor(final_thresh, cv2.COLOR_BGR2GRAY)


def kmeans_clustering(preprocessed_img, k):
    # Reshape image
    pixels = preprocessed_img.reshape(-1, 1).astype('float32')

    # Instantiate the index
    d = pixels.shape[1]  # Dimension of the data points
    index = faiss.IndexFlatL2(d)

    # Train the index
    niter = 200
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = niter
    clus.train(pixels, index)

    # Get the centroids
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(k, d)

    # Assign each pixel to the nearest centroid
    _, labels = index.search(pixels, 1)

    # Reshape labels to match original image shape
    clustered_img = labels.reshape(preprocessed_img.shape)

    # Count the number of pixels for each label
    unique_labels, label_counts = np.unique(clustered_img, return_counts=True)

    # Find the label with the minimum size
    min_size_label = unique_labels[np.argmin(label_counts)]

    # Create binary image with only the smallest label
    smallest_label_img = np.where(clustered_img == min_size_label, 255, 0).astype(np.uint8)

    return smallest_label_img, centroids


def morphology_operation(img):
    # Define structuring element (disk with radius 3)
    selem = disk(3)

    # Erosion followed by dilation
    eroded_img = binary_erosion(img, selem)
    morph_img = binary_dilation(eroded_img, selem)

    return morph_img


def binary_segmentation(original_img, binary_img):
    # Convert binary image to the appropriate data type for the mask
    binary_img = binary_img.astype(np.uint8)

    # Use bitwise_and with mask
    segmented_img = cv2.bitwise_and(original_img, original_img, mask=binary_img)
    return segmented_img


def main(args):
    # Read original image
    original_img = cv2.imread(args.input_image)

    # Preprocess image
    preprocessed_img = preprocess_image(original_img)

    # K-means clustering
    kmeans_img, centroid = kmeans_clustering(preprocessed_img, args.num_clusters)

    # Morphology operation
    morph_img = morphology_operation(kmeans_img)

    # Binary segmentation
    segmented_img_thresh = binary_segmentation(original_img, preprocessed_img)
    segmented_img_kmeans = binary_segmentation(original_img, kmeans_img)
    segmented_img_morph = binary_segmentation(original_img, morph_img)

    # For tracing
    if args.vis:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(preprocessed_img, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')

        axes[0, 2].imshow(morph_img, cmap='gray')
        axes[0, 2].set_title('Morphology')

        axes[1, 0].imshow(cv2.cvtColor(segmented_img_thresh, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Segmented Image (Thresholding)')

        axes[1, 1].imshow(cv2.cvtColor(segmented_img_kmeans, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Segmented Image (K-means)')

        axes[1, 2].imshow(cv2.cvtColor(segmented_img_morph, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Segmented Image (Morphology)')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    # saving
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save output images independently if specified

    # cv2.imwrite(os.path.join(output_dir, 'segmented_img_thresh.jpg'), segmented_img_thresh)
    # cv2.imwrite(os.path.join(output_dir, 'segmented_img_kmeans.jpg'), segmented_img_kmeans)
    cv2.imwrite(os.path.join(output_dir, 'segmented_img_morph.jpg'), segmented_img_morph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image segmentation using thresholding and K-means clustering.")
    parser.add_argument("input_image", help="Path to the input image file.")
    parser.add_argument("--threshold1", type=int, default=93, help="First threshold value.")
    parser.add_argument("--threshold2", type=int, default=110, help="Second threshold value.")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters for K-means clustering.")
    parser.add_argument("--vis", action="store_true", help="Plot all phases using thresholding and K-means clustering.")
    parser.add_argument("--output_dir", type=str, default='output', help="Save the masks into a specific path. If the path does not exist, the path will be created by default.")
    args = parser.parse_args()

    main(args)
