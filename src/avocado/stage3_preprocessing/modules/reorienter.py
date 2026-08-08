import cv2
import numpy as np


def reorient_fruit(mask: np.ndarray, crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorients the fruit vertically with the smaller peak at the top.
    Centers it in a new padded frame.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, crop_bgr
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) == 0:
        return mask, crop_bgr

    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        angle = ellipse[2]
    else:
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90

    # Ensure angle brings fruit closer to vertical
    rotation_angle = angle
    if rotation_angle > 90:
        rotation_angle -= 180

    h, w = mask.shape
    diag = int(np.ceil(np.sqrt(w*w + h*h)))
    
    pad_w = (diag - w) // 2
    pad_h = (diag - h) // 2
    
    padded_mask = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    padded_crop = cv2.copyMakeBorder(crop_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    
    # Rotate
    M_rot = cv2.getRotationMatrix2D((diag//2, diag//2), rotation_angle, 1.0)
    rotated_mask = cv2.warpAffine(padded_mask, M_rot, (diag, diag), flags=cv2.INTER_NEAREST)
    rotated_crop = cv2.warpAffine(padded_crop, M_rot, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Check if smaller peak is at the top or bottom
    rx, ry, rw, rh = cv2.boundingRect(rotated_mask)
    if rh > 0 and rw > 0:
        top_half = rotated_mask[ry:ry + rh//2, rx:rx + rw]
        bottom_half = rotated_mask[ry + rh//2:ry + rh, rx:rx + rw]
        
        top_area = cv2.countNonZero(top_half)
        bottom_area = cv2.countNonZero(bottom_half)
        
        if top_area > bottom_area:
            # The larger part is at the top. Flip 180 degrees so the smaller peak is at the top.
            M_rot180 = cv2.getRotationMatrix2D((diag//2, diag//2), 180, 1.0)
            rotated_mask = cv2.warpAffine(rotated_mask, M_rot180, (diag, diag), flags=cv2.INTER_NEAREST)
            rotated_crop = cv2.warpAffine(rotated_crop, M_rot180, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Center it in a tight crop with slight padding
    final_rx, final_ry, final_rw, final_rh = cv2.boundingRect(rotated_mask)
    if final_rw > 0 and final_rh > 0:
        pad = 20
        centered_mask = rotated_mask[
            max(0, final_ry-pad) : min(diag, final_ry+final_rh+pad), 
            max(0, final_rx-pad) : min(diag, final_rx+final_rw+pad)
        ]
        centered_crop = rotated_crop[
            max(0, final_ry-pad) : min(diag, final_ry+final_rh+pad), 
            max(0, final_rx-pad) : min(diag, final_rx+final_rw+pad)
        ]
        return centered_mask, centered_crop
    
    return rotated_mask, rotated_crop
