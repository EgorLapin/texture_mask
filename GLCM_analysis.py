"""
GLCM texture analysis for skimage.data 'brick' and 'camera'.

Features computed: contrast, energy, homogeneity.
Saves a human-readable TXT report and shows images side-by-side.

Raises:
    Exception: For various errors during processing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


def load_image(source):
    """Load a sample image by name.

    Args:
        source: Name of the image to load. Supported values: "camera", "brick".

    Returns:
        ndarray: Loaded image array (grayscale or RGB) or None on error.

    Raises:
        ValueError: If an unsupported source name is provided.
        Exception: For other errors during loading.
    """
    try:
        if source == "camera":
            return data.camera()
        elif source == "brick":
            return data.brick()
        else:
            raise ValueError("Unsupported image source. Use 'camera' or 'brick'.")
    except Exception as exc:
        print(f"[load_image] Error loading '{source}': {exc}")
        return None


def prepare_gray(img):
    """Convert image to grayscale uint8.

    The function converts RGB -> luminance grayscale and ensures dtype is uint8
    with values in the range 0..255.

    Args:
        img: Input image array.

    Returns:
        ndarray: Grayscale uint8 image or None on error.

    Raises:
        ValueError: If img is None or invalid.
        Exception: For other errors during conversion.
    """
    try:
        if img is None:
            raise ValueError("Input image is None.")

        if img.ndim == 3:
            img = color.rgb2gray(img) # Convert RGB to grayscale luminance if necessary

        img_u8 = img_as_ubyte(img) # Convert to uint8 [0..255]
        return img_u8

    except Exception as exc:
        print(f"[prepare_gray] Error: {exc}")
        return None


def compute_glcm(gray_img, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=256):
    """Compute a normalized symmetric Gray Level Co-occurrence Matrix (GLCM).

    Args:
        gray_img: Grayscale uint8 image (0..255).
        distances: Iterable of pixel-pair distances.
        angles: Iterable of offsets (radians).
        levels: Number of gray levels to consider (default 256 for uint8).

    Returns:
        ndarray: GLCM array (or None on error).

    Raises:
        ValueError: If gray_img is None or not uint8.
        Exception: For other errors during GLCM computation.
    """
    try:
        if gray_img is None:
            raise ValueError("gray_img is None.")
        if gray_img.dtype != np.uint8:
            raise ValueError("gray_img must be uint8 (0..255). Use prepare_gray() first.")

        glcm = graycomatrix(gray_img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        return glcm
    except Exception as exc:
        print(f"[compute_glcm] Error: {exc}")
        return None


def metric_contrast(glcm):
    """Compute the contrast metric from the GLCM.

    Contrast measures local intensity variation: higher values indicate stronger local differences.

    Args:
        glcm: Gray-level co-occurrence matrix.

    Returns:
        float: Mean contrast across provided distances and angles (NaN on error).

    Raises:
        ValueError: If glcm is None.
        Exception: For other errors during metric calculation.
    """
    try:
        if glcm is None:
            raise ValueError("GLCM is None.")
        vals = graycoprops(glcm, 'contrast')
        return float(np.mean(vals))
    except Exception as exc:
        print(f"[metric_contrast] Error: {exc}")
        return float('nan')


def metric_energy(glcm):
    """Compute the energy metric from the GLCM.

    Energy (Angular Second Moment) is a measure of uniformity. Larger values indicate more order.

    Args:
        glcm: Gray-level co-occurrence matrix.

    Returns:
        float: Mean energy across distances and angles (NaN on error).

    Raises:
        ValueError: If glcm is None.
        Exception: For other errors during metric calculation.
    """
    try:
        if glcm is None:
            raise ValueError("GLCM is None.")
        vals = graycoprops(glcm, 'energy')
        return float(np.mean(vals))
    except Exception as exc:
        print(f"[metric_energy] Error: {exc}")
        return float('nan')


def metric_homogeneity(glcm):
    """Compute the homogeneity metric from the GLCM.

    Homogeneity measures closeness of element distribution to the GLCM diagonal (smoothness).

    Args:
        glcm: Gray-level co-occurrence matrix.

    Returns:
        float: Mean homogeneity across distances and angles (NaN on error).

    Raises:
        ValueError: If glcm is None.
        Exception: For other errors during metric calculation.
    """
    try:
        if glcm is None:
            raise ValueError("GLCM is None.")
        vals = graycoprops(glcm, 'homogeneity')
        return float(np.mean(vals))
    except Exception as exc:
        print(f"[metric_homogeneity] Error: {exc}")
        return float('nan')


def compute_glcm_features(img, name, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Prepare image, compute GLCM with levels=256 and extract texture features.

    Args:
        img: Input image (RGB or grayscale).
        name: Human-friendly image name (used in outputs).
        distances: Distances to use for GLCM.
        angles: Angles to use for GLCM.

    Returns:
        tuple: (name, features_dict, glcm) where features_dict contains contrast, energy, homogeneity.

    Raises:
        RuntimeError: If grayscale preparation or GLCM computation fails.
        Exception: For other errors during feature extraction.
    """
    try:
        gray = prepare_gray(img) # Convert to grayscale uint8
        if gray is None:
            raise RuntimeError("Failed to prepare grayscale image.")

        glcm = compute_glcm(gray, distances=distances, angles=angles, levels=256) # Compute GLCM using full uint8 range
        if glcm is None:
            raise RuntimeError("Failed to compute GLCM.")

        # Compute metrics using dedicated functions
        contrast = metric_contrast(glcm)
        energy = metric_energy(glcm)
        homogeneity = metric_homogeneity(glcm)

        features = {
            "contrast": contrast,
            "energy": energy,
            "homogeneity": homogeneity,
        }
        return name, features, glcm

    except Exception as exc:
        print(f"[compute_glcm_features] Error for '{name}': {exc}")
        return name, {}, None


def display_image(img, title="image"):
    """Display a single grayscale image.

    Args:
        img: Input image (RGB or grayscale).
        title: Plot title.

    Raises:
        RuntimeError: If image cannot be displayed.
        Exception: For other errors during display.
    """
    try:
        gray = prepare_gray(img)
        if gray is None:
            raise RuntimeError("Cannot display empty image.")
        plt.imshow(gray, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
    except Exception as exc:
        print(f"[display_image] Error: {exc}")


def display_two_images(img1, img2, title1="img1", title2="img2"):
    """Display two images side-by-side for visual comparison.

    Args:
        img1: First image.
        img2: Second image.
        title1: Title for first image.
        title2: Title for second image.

    Raises:
        RuntimeError: If either image cannot be displayed.
        Exception: For other errors during display.
    """
    try:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        gray1 = prepare_gray(img1)
        if gray1 is None:
            raise RuntimeError("First image cannot be displayed.")
        plt.imshow(gray1, cmap="gray")
        plt.title(title1)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        gray2 = prepare_gray(img2)
        if gray2 is None:
            raise RuntimeError("Second image cannot be displayed.")
        plt.imshow(gray2, cmap="gray")
        plt.title(title2)
        plt.axis("off")

        plt.show()
    except Exception as exc:
        print(f"[display_two_images] Error: {exc}")


def compare_and_print(res1, res2):
    """Compare feature dictionaries, print readable lines and return comparison info.

    Args:
        res1: Tuple (name, features, glcm) for image A.
        res2: Tuple (name, features, glcm) for image B.

    Returns:
        dict: Comparison info for each metric: {'contrast': {'a':..., 'b':..., 'sign':..., 'diff':...}, ...}

    Raises:
        Exception: For errors during comparison.
    """
    try:
        name1, feats1, _ = res1
        name2, feats2, _ = res2

        metrics = ["contrast", "energy", "homogeneity"]
        comparison = {}

        print(f"\nFeature values for '{name1}' and '{name2}':")
        for m in metrics:
            v1 = feats1.get(m, float('nan'))
            v2 = feats2.get(m, float('nan'))

            if np.isnan(v1) or np.isnan(v2):
                sign = "?"
                diff = None
            else:
                sign = ">" if v1 > v2 else "<" if v1 < v2 else "="
                diff = float(v1 - v2)

            v1_str = f"{v1:.4f}" if not np.isnan(v1) else "nan"
            v2_str = f"{v2:.4f}" if not np.isnan(v2) else "nan"
            print(f" {m}: {name1} ({v1_str}) {sign} {name2} ({v2_str})")

            comparison[m] = {"a": v1, "b": v2, "sign": sign, "diff": diff}

        return comparison

    except Exception as exc:
        print(f"[compare_and_print] Error: {exc}")
        return {}


def save_results_txt(res1, res2, comparison, filename="results.txt"):
    """Save a human-readable comparison report to a TXT file.

    Args:
        res1: Tuple (name, features, glcm) for image A.
        res2: Tuple (name, features, glcm) for image B.
        comparison: Comparison dictionary produced by compare_and_print.
        filename: Output TXT filename.

    Raises:
        Exception: For errors during file writing.
    """
    try:
        name1, feats1, _ = res1
        name2, feats2, _ = res2

        lines = []
        lines.append("GLCM texture analysis report")
        lines.append("============================")
        lines.append("")
        lines.append(f"Image A: {name1}")
        for k, v in feats1.items():
            # Format numeric values neatly where possible
            if isinstance(v, (float, int)) and not np.isnan(v):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"Image B: {name2}")
        for k, v in feats2.items():
            if isinstance(v, (float, int)) and not np.isnan(v):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Comparison:")
        for metric, comp in comparison.items():
            a = comp.get("a", "nan")
            b = comp.get("b", "nan")
            sign = comp.get("sign", "?")
            diff = comp.get("diff", None)
            diff_str = f"{diff:.6f}" if diff is not None else "N/A"
            lines.append(f"  {metric}: {name1} ({a}) {sign} {name2} ({b}) | diff = {diff_str}")
        lines.append("")

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True) # Ensure directory exists

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        print(f"Saved TXT report to: {filename}")
    except Exception as exc:
        print(f"[save_results_txt] Error saving TXT: {exc}")


def save_result_images(img1, img2, title1="img1", title2="img2", filename="result_images.png"):
    """Save two images side-by-side to disk.

    Args:
        img1: First image.
        img2: Second image.
        title1: Title for first image.
        title2: Title for second image.
        filename: Output image filename.

    Raises:
        Exception: For errors during image saving.
    """
    try:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        gray1 = prepare_gray(img1)
        if gray1 is None:
            raise RuntimeError("First image cannot be saved.")
        plt.imshow(gray1, cmap="gray")
        plt.title(title1)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        gray2 = prepare_gray(img2)
        if gray2 is None:
            raise RuntimeError("Second image cannot be saved.")
        plt.imshow(gray2, cmap="gray")
        plt.title(title2)
        plt.axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved image comparison to: {filename}")
    except Exception as exc:
        print(f"[save_result_images] Error saving images: {exc}")


def execute(save_txt="results.txt", save_img="result_images.png"):
    """Main pipeline: load images, compute features, compare, display and save TXT report.

    Args:
        save_txt: Output TXT filename for the human-readable report.
        save_img: Output image filename for the side-by-side comparison.

    Raises:
        RuntimeError: If image loading fails.
        Exception: For other errors during execution.
    """
    try:
        # Load images
        img_brick = load_image("brick")
        img_camera = load_image("camera")
        if img_brick is None or img_camera is None:
            raise RuntimeError("Failed to load one or more images.")

        # Compute features for both images
        res_brick = compute_glcm_features(img_brick, "brick")
        res_camera = compute_glcm_features(img_camera, "camera")

        # Print per-image features to console
        for name, feats, _ in (res_brick, res_camera):
            print(f"\n{name} features:")
            for k, v in feats.items():
                try:
                    v_str = f"{v:.6f}" if isinstance(v, (float, int)) and not np.isnan(v) else str(v)
                except Exception:
                    v_str = str(v)
                print(f"  {k}: {v_str}")

        # Compare and print comparison
        comparison = compare_and_print(res_brick, res_camera)

        # Show images side by side for visual inspection
        display_two_images(img_brick, img_camera, title1="brick", title2="camera")

        # Save only TXT report
        save_results_txt(res_brick, res_camera, comparison, filename=save_txt)

        # Save result images side-by-side
        save_result_images(img_brick, img_camera, title1="brick", title2="camera", filename=save_img)

    except Exception as exc:
        print(f"[execute] Fatal error: {exc}")

if __name__ == "__main__":
    try:
        execute()
    except Exception as e:
        print(f"Fatal error: {e}")
