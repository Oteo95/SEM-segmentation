import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import random
import logging
from patchify import patchify
from datasets import Dataset
from PIL import Image


# Constants
DATA_DIR = "../..//data/mitochondria"
PATCH_SIZE = 256
STEP_SIZE = 256

def load_tiff_images(file_path: str) -> np.ndarray:
    """Load tiff images from the given file path."""
    try:
        return tifffile.imread(file_path)
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise

def create_patches(image_stack: np.ndarray, patch_size: int, step: int) -> np.ndarray:
    """Create patches from the image stack."""
    patches = []
    for img in range(image_stack.shape[0]):
        patches_img = patchify(image_stack[img], (patch_size, patch_size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                patches.append(patches_img[i, j, :, :])
    return np.array(patches)

def create_mask_patches(image_stack: np.ndarray, patch_size: int, step: int) -> np.ndarray:
    """Create patches from the image stack."""
    patches = []
    for img in range(image_stack.shape[0]):
        patches_img = patchify(image_stack[img], (patch_size, patch_size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_mask = patches_img[i, j, :, :]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                patches.append(single_patch_mask)
    return np.array(patches)

def filter_non_empty_masks(images: np.ndarray, masks: np.ndarray) -> tuple:
    """Filter out images and masks that are empty."""
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    return images[valid_indices], masks[valid_indices]

def convert_to_dataset(images: np.ndarray, masks: np.ndarray) -> Dataset:
    """Convert NumPy arrays of images and masks to a dataset."""
    dataset_dict = {
        "image": [Image.fromarray(img) for img in images],
        "label": [Image.fromarray(mask) for mask in masks],
    }
    return Dataset.from_dict(dataset_dict)

def plot_examples(dataset: Dataset, num_examples: int = 1):
    """Plot example images and masks from the dataset."""
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))
    for i in range(num_examples):
        img_num = random.randint(0, len(dataset) - 1)
        example_image = dataset[img_num]["image"]
        example_mask = dataset[img_num]["label"]
        #print(axes[0], axes[1])
        plt.imshow(np.array(example_image), cmap='gray')
        axes[i].imshow(np.array(example_image), cmap='gray')
        axes[i].set_title("Image")
        axes[i].imshow(np.array(example_mask), cmap='gray')
        axes[i].set_title("Mask")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.show()


# Main Script
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load and process images
    large_images = load_tiff_images("/workspaces/SEM-segmentation/data/mitochondria/training.tif")[:10,:,:]
    large_masks = load_tiff_images("/workspaces/SEM-segmentation/data/mitochondria/training_groundtruth.tif")[:10,:,:]
    images_patches = create_patches(large_images, PATCH_SIZE, STEP_SIZE)
    masks_patches = create_mask_patches(large_masks, PATCH_SIZE, STEP_SIZE)
    filtered_images, filtered_masks = filter_non_empty_masks(images_patches, masks_patches)
    dataset = convert_to_dataset(filtered_images, filtered_masks)

    # Plot examples
    plot_examples(dataset)
