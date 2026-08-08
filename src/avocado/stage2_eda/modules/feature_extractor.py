import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import os

def extract_features(image_path, fruit_id, timestamp, config=None):
    """
    Extracts shape, color, and texture features from a segmented avocado image.
    Uses the alpha channel as the fruit mask.
    """
    if config is None:
        config = {}
        
    dark_threshold_l = config.get("dark_threshold_l", 35) # Lightness threshold in LAB space
    min_dark_component_size = config.get("min_dark_component_size", 50) # To filter out noise
    glcm_levels = config.get("glcm_levels", 32)
    
    # Initialize feature dict
    features = {
        "fruit_id": fruit_id,
        "timestamp": timestamp,
        "image_path": image_path,
        "valid": False
    }

    # Load image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        return features # Invalid or no alpha channel
        
    b, g, r, a = cv2.split(img)
    mask = a > 0
    
    # Shape & Segmentation Metrics
    mask_area = np.sum(mask)
    if mask_area == 0:
        return features
        
    features["valid"] = True
    features["mask_area"] = mask_area
    
    # Bounding box
    y_indices, x_indices = np.where(mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    features["bounding_box_area"] = bbox_area
    features["mask_bbox_ratio"] = mask_area / bbox_area if bbox_area > 0 else 0
    features["fruit_width"] = x_max - x_min + 1
    features["fruit_height"] = y_max - y_min + 1
    features["centroid_x"] = np.mean(x_indices)
    features["centroid_y"] = np.mean(y_indices)
    
    total_pixels = img.shape[0] * img.shape[1]
    features["transparent_pixel_ratio"] = (total_pixels - mask_area) / total_pixels
    
    # Color Metrics
    r_masked = r[mask]
    g_masked = g[mask]
    b_masked = b[mask]
    
    features["mean_r"] = np.mean(r_masked)
    features["mean_g"] = np.mean(g_masked)
    features["mean_b"] = np.mean(b_masked)
    features["median_r"] = np.median(r_masked)
    features["median_g"] = np.median(g_masked)
    features["median_b"] = np.median(b_masked)
    
    # Green Ratio (G / (R+G+B)) and Excess Green (2G - R - B)
    rgb_sum = r_masked.astype(float) + g_masked.astype(float) + b_masked.astype(float) + 1e-6
    features["green_ratio"] = np.mean(g_masked.astype(float) / rgb_sum)
    features["excess_green"] = np.mean(2 * g_masked.astype(float) - r_masked.astype(float) - b_masked.astype(float))
    
    # HSV and LAB
    bgr = cv2.merge([b, g, r])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    h, s, v = cv2.split(hsv)
    l_chan, a_chan, b_chan = cv2.split(lab)
    
    features["mean_hue"] = np.mean(h[mask])
    features["mean_saturation"] = np.mean(s[mask])
    features["mean_value"] = np.mean(v[mask])
    
    features["mean_lab_l"] = np.mean(l_chan[mask])
    features["mean_lab_a"] = np.mean(a_chan[mask])
    features["mean_lab_b"] = np.mean(b_chan[mask])
    
    # Dark coverage (filtering small components)
    # Using L channel for darkness threshold
    dark_mask = (l_chan < dark_threshold_l) & mask
    
    # Connected components to filter noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8), connectivity=8)
    filtered_dark_mask = np.zeros_like(dark_mask)
    
    largest_dark_spot = 0
    dark_component_count = 0
    
    for i in range(1, num_labels): # Skip 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_dark_component_size:
            filtered_dark_mask[labels == i] = True
            largest_dark_spot = max(largest_dark_spot, stats[i, cv2.CC_STAT_AREA])
            dark_component_count += 1
            
    filtered_dark_area = np.sum(filtered_dark_mask)
    features["dark_coverage"] = filtered_dark_area / mask_area if mask_area > 0 else 0
    features["dark_component_count"] = dark_component_count
    features["largest_dark_spot_fraction"] = largest_dark_spot / mask_area if mask_area > 0 else 0
    
    # Texture Metrics (Entropy, GLCM, Edges)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Entropy on masked region only
    features["texture_entropy"] = shannon_entropy(gray[mask])
    
    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = np.sum(edges[mask] > 0) / mask_area if mask_area > 0 else 0
    
    # Laplacian variance (sharpness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["laplacian_variance"] = np.var(laplacian[mask])
    
    # GLCM (Quantize grayscale first)
    # Extract bounding box to reduce background computation
    gray_bbox = gray[y_min:y_max+1, x_min:x_max+1]
    
    # Quantize
    quantized_gray = (gray_bbox.astype(np.float32) / 256.0 * glcm_levels).astype(np.uint8)
    
    glcm = graycomatrix(quantized_gray, distances=[1], angles=[0], levels=glcm_levels, symmetric=True, normed=True)
    features["glcm_contrast"] = graycoprops(glcm, 'contrast')[0, 0]
    features["glcm_homogeneity"] = graycoprops(glcm, 'homogeneity')[0, 0]

    return features
