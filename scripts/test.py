import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_to_canny(image_path, low_threshold=100, high_threshold=200):
    """
    Load an RGB or RGBA image, convert it to grayscale, and compute Canny edges.

    Args:
        image_path (str): Path to the input image.
        low_threshold (int): Lower threshold for Canny.
        high_threshold (int): Upper threshold for Canny.

    Returns:
        original_rgb (np.ndarray): Original image in RGB format.
        edges (np.ndarray): Canny edge image.
    """
    # Read with unchanged mode so RGB / RGBA are both preserved
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Handle grayscale, BGR, and BGRA
    if len(img.shape) == 2:
        original_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gray = img
    elif img.shape[2] == 3:
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # For Canny, use RGB channels only
        rgb_for_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(rgb_for_gray, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return original_rgb, edges


if __name__ == "__main__":
    image_path = "./xray.png"  # change this to your image path

    original, canny = image_to_canny(
        image_path=image_path,
        low_threshold=100,
        high_threshold=200,
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(canny, cmap="gray")
    plt.title("Canny Edge Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
