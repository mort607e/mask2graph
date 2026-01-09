from typing import Tuple
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_holes

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from preprocess.patch_preprocessing import open_tif


def binaryMaskCheck(mask: np.ndarray) -> None:
    """
    Checks if the provided mask is a valid binary mask.
    
    Args:
    mask: np.ndarray (uint8)

    Returns:
    None
    """
    if mask.dtype != np.uint8:
        raise ValueError(f"Invalied data type {mask.dtype} expected np.uint8")
    
    if len(mask.shape) != 2:
        raise ValueError(f"Invalid mask shape: {mask.shape}. Mask must be 2-dimensional.")
    
    
    if mask.size == 0:
        raise ValueError("Mask is empty. Please provide a non-empty mask.")

    unique_values = np.unique(mask)
    if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [0, 1]):
        raise ValueError(f"Mask contains invalid values: {unique_values}. Only binary values (0 and 1) are allowed.")

def construct_kernel(shape: int, ksize: Tuple[int, int]) -> np.ndarray:
    """
    Constructs a structuring element (kernel) for morphological operations.

    Args:
    shape: int
        The shape of the structuring element. For example, cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS.
    ksize: Tuple[int, int]
        The size of the kernel (width, height).

    Returns:
    np.ndarray
        The constructed structuring element.
    """
    return cv2.getStructuringElement(shape, ksize)

def morphological_opening(mask: np.ndarray, kernel: np.array) -> np.ndarray:
    """
    Apply morphological opening to mask

    Args:
    mask: np.ndarray
    kernel: np.ndarray

    Return 
    np.ndarray
    """
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def morphological_closening(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply morphological closening to mask

    Args:
    mask: np.ndarray
    kernel: np.ndarray

    Return 
    np.ndarray
    """
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def filter_components_by_area_ratio(mask: np.ndarray, max_area_ratio: float = 0.2, connectivity: int = 8) -> np.ndarray:
    
    """
    Compute an area ratio for each component found in the mask, removes all components with an area ratio greater than max_area_ratio.

    Args:
    mask: np.ndarray
    max_area_ratio: float
    connectivity: int

    Returns:
    np.ndarray
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    clean_mask = np.zeros_like(mask)

    for label in range(1, num_labels):  # Skip background label (0)
        component_area = stats[label, cv2.CC_STAT_AREA]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]

        area_ratio = component_area/(component_width*component_height)

        if area_ratio < max_area_ratio:
            clean_mask[labels == label] = 1

    return clean_mask

def fill_small_holes(mask: np.ndarray, min_area: int = 1, *args, **kwargs) -> np.ndarray:
    
    """
    Removes all holes (pixels in a neighbourhood) smaller than the specified area.

    Args:
    mask: np.ndarray (uint8)
    min_area: int
        Minimum area of holes to be filled.
    *args: Additional positional arguments passed to `remove_small_holes`.
    **kwargs: Additional keyword arguments passed to `remove_small_holes`.


    Returns:
    np.ndarray
        Mask with small holes filled.
    """

    filled_mask= remove_small_holes(mask.astype(bool), area_threshold=min_area, *args, **kwargs)
    return filled_mask.astype(np.uint8)
    

if __name__ == '__main__':
    # For plot of the masks
    import matplotlib.pyplot as plt

    tif_path = r"data\sample\Predicted_Mask_Bramaputra_2020-03-02.tif"
    result = open_tif(tif_path)
    mask: np.ndarray = result[0][0]
    mask = mask.astype(np.uint8)

    # Example:
    kernel = construct_kernel(cv2.MORPH_RECT, (5,5))

    binaryMaskCheck(mask)

    print("Applying morphological operations:")
    print("Opening")
    morph_opened_mask = morphological_opening(mask, kernel)
    print("Closening")
    morph_closed_mask = morphological_closening(morph_opened_mask, kernel)

    print("Filtering components by area ratio:")
    filtered_mask = filter_components_by_area_ratio(morph_closed_mask, max_area_ratio=0.15)

    print("Removing holes:")
    filled_mask = fill_small_holes(filtered_mask, 25000) #exaggeration
    print("Done")

    plt.figure(figsize=(16, 4))

    # Original mask
    plt.subplot(1, 4, 1)
    plt.title("Original Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # After morphological operations
    plt.subplot(1, 4, 2)
    plt.title("Morphological Operations")
    plt.imshow(morph_closed_mask, cmap='gray')
    plt.axis('off')

    # Final filtered mask
    plt.subplot(1, 4, 3)
    plt.title("Filtered Mask")
    plt.imshow(filtered_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Filled Mask (exaggeration)")
    plt.imshow(filled_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(r"images\CleanedMask.png")
    plt.show()


