import cv2
import numpy as np
import os
import glob
import re
from pathlib import Path

# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "01_raw" / "avocado" / "output"
MASK_DIR = PROJECT_ROOT / "data" / "excluded_masks"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def load_mask(path, target_shape):
    """
    Loads mask from a PNG file. Handles alpha channels, transparent backgrounds,
    and different image formats by auto-detecting the foreground polarity.
    """
    if not path.exists():
        return np.zeros(target_shape, dtype=np.uint8)
        
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros(target_shape, dtype=np.uint8)
        
    # Resize to target shape if mismatch
    if img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Determine binary mask based on channels
    if len(img.shape) == 3 and img.shape[2] == 4:
        # 4 Channels (RGBA)
        alpha = img[:, :, 3]
        # If alpha has variations, use it as a mask
        if np.any(alpha < 255) and np.any(alpha > 0):
            mask = (alpha > 50).astype(np.uint8) * 255
        else:
            # Fall back to RGB color thresholding
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        # 3 Channels (BGR) or 1 Channel (Grayscale)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Check background polarity
        num_white = np.sum(gray > 240)
        num_black = np.sum(gray < 15)
        if num_white > num_black:
            # Light background -> dark pixels are mask
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            # Dark background -> light pixels are mask
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            
    return mask

def main():
    print("=" * 60)
    print("      INTERACTIVE MASK VISUALIZATION PLAYER")
    print("=" * 60)
    print(f"Image Directory: {RAW_IMAGE_DIR}")
    print(f"Mask Directory:  {MASK_DIR}")
    
    # 1. Collect raw images
    image_paths = sorted(list(RAW_IMAGE_DIR.glob("*.jpg")), key=natural_sort_key)
    if not image_paths:
        print(f"ERROR: No .jpg files found in {RAW_IMAGE_DIR}")
        return
    print(f"Found {len(image_paths)} raw images.")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        print(f"ERROR: Cannot read the first image: {image_paths[0]}")
        return
    h, w = first_img.shape[:2]
    target_shape = (h, w)
    print(f"Resolution: {w}x{h}")
    
    # 2. Load masks
    mask_files = {
        1: MASK_DIR / "ex_object_1.drawio.png",
        2: MASK_DIR / "ex_object_2.drawio.png",
        3: MASK_DIR / "ex_object_3.drawio.png"
    }
    
    masks = {}
    for key, path in mask_files.items():
        if path.exists():
            masks[key] = load_mask(path, target_shape)
            print(f"Loaded Mask {key}: {path.name} (pixels active: {np.count_nonzero(masks[key])})")
        else:
            masks[key] = np.zeros(target_shape, dtype=np.uint8)
            print(f"WARNING: Mask file not found, initializing empty: {path}")
            
    # Combine masks
    combined_mask = cv2.bitwise_or(masks[1], cv2.bitwise_or(masks[2], masks[3]))
    print(f"Full Combined Mask active pixels: {np.count_nonzero(combined_mask)}")
    
    # Visual states
    # Mode 0: Clean image
    # Mode 1: Mask 1 (Red)
    # Mode 2: Mask 2 (Green)
    # Mode 3: Mask 3 (Blue)
    # Mode 4: Full/Combined Mask (Purple)
    # Mode 5: All Separated Masks (Red, Green, Blue layered)
    mode = 5  # Start with all separated masks by default
    mode_names = {
        0: "Clean Image (No Overlays)",
        1: "Mask 1 Only (ex_object_1)",
        2: "Mask 2 Only (ex_object_2)",
        3: "Mask 3 Only (ex_object_3)",
        4: "Full Combined Mask",
        5: "All Separated Masks (R=Obj1, G=Obj2, B=Obj3)"
    }
    
    # Playback settings
    current_idx = 0
    is_playing = False
    delay = 100  # ms delay between frames (approx 10 FPS)
    
    window_name = "Mask Visualization Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("\nControls:")
    print("  [Space]     - Play / Pause")
    print("  [D] / [->]  - Next frame (when paused)")
    print("  [A] / [<-]  - Previous frame (when paused)")
    print("  [1] - [3]   - View individual Object Mask 1, 2, or 3")
    print("  [4]         - View Combined/Full Mask")
    print("  [5]         - View All Separated Masks simultaneously in different colors")
    print("  [0] / [C]   - Clear overlay (show original frame)")
    print("  [+] / [W]   - Increase play speed")
    print("  [-] / [S]   - Decrease play speed")
    print("  [Q] / [ESC] - Quit")
    
    while True:
        # Load frame
        img_path = image_paths[current_idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Error loading frame index {current_idx}: {img_path.name}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue
            
        display_frame = frame.copy()
        
        # Apply mask overlays
        overlay = np.zeros_like(frame)
        
        if mode == 1:
            overlay[masks[1] > 0] = [0, 0, 255]  # Red
            cv2.addWeighted(overlay, 0.45, display_frame, 1.0, 0, display_frame)
        elif mode == 2:
            overlay[masks[2] > 0] = [0, 255, 0]  # Green
            cv2.addWeighted(overlay, 0.45, display_frame, 1.0, 0, display_frame)
        elif mode == 3:
            overlay[masks[3] > 0] = [255, 0, 0]  # Blue
            cv2.addWeighted(overlay, 0.45, display_frame, 1.0, 0, display_frame)
        elif mode == 4:
            overlay[combined_mask > 0] = [255, 0, 255]  # Magenta
            cv2.addWeighted(overlay, 0.45, display_frame, 1.0, 0, display_frame)
        elif mode == 5:
            # Color code each mask separately
            overlay[masks[1] > 0] = [0, 0, 255]   # Mask 1 = Red
            overlay[masks[2] > 0] = [0, 255, 0]   # Mask 2 = Green
            overlay[masks[3] > 0] = [255, 0, 0]   # Mask 3 = Blue
            
            # Combine cleanly (prevent overlap color muddiness)
            mask_rgb_active = (masks[1] > 0) | (masks[2] > 0) | (masks[3] > 0)
            cv2.addWeighted(overlay, 0.45, display_frame, 1.0, 0, display_frame)
            
        # Draw HUD info
        hud_top_height = 55
        hud_bottom_height = 45
        
        # Top HUD semi-transparent background
        hud_top = display_frame[0:hud_top_height, 0:w].copy()
        cv2.rectangle(hud_top, (0, 0), (w, hud_top_height), (0, 0, 0), -1)
        cv2.addWeighted(hud_top, 0.65, display_frame[0:hud_top_height, 0:w], 0.35, 0, display_frame[0:hud_top_height, 0:w])
        
        # Bottom HUD semi-transparent background
        hud_bottom = display_frame[h-hud_bottom_height:h, 0:w].copy()
        cv2.rectangle(hud_bottom, (0, 0), (w, hud_bottom_height), (0, 0, 0), -1)
        cv2.addWeighted(hud_bottom, 0.65, display_frame[h-hud_bottom_height:h, 0:w], 0.35, 0, display_frame[h-hud_bottom_height:h, 0:w])
        
        # Draw status text
        status_str = "PLAYING" if is_playing else "PAUSED"
        status_color = (0, 255, 0) if is_playing else (0, 255, 255)
        
        # Top HUD Text
        text_y_offset = 35
        cv2.putText(display_frame, f"Frame: {current_idx + 1}/{len(image_paths)} | {img_path.name}", 
                    (20, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Right aligned status
        status_text = f"Status: {status_str} | Speed: {delay}ms"
        (status_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(display_frame, status_text, 
                    (w - status_w - 20, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
        
        # Bottom HUD Text (Shortcuts and Mode)
        mode_text = f"Mode: {mode_names[mode]}"
        cv2.putText(display_frame, mode_text, 
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        shortcuts_text = "Space: Play/Pause | A/D: Prev/Next | 1-5: Choose Mask Overlay | 0: Hide Mask | Q: Quit"
        (sh_w, _), _ = cv2.getTextSize(shortcuts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(display_frame, shortcuts_text, 
                    (w - sh_w - 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Display image
        cv2.imshow(window_name, display_frame)
        
        # Wait for keypress
        wait_time = delay if is_playing else 0
        raw_key = cv2.waitKeyEx(wait_time)
        
        # Handle OpenCV waitKey variations
        key = raw_key & 0xFF if raw_key != -1 else -1
        
        # Key commands
        if raw_key == 27 or key == ord('q') or key == ord('Q'):  # Esc or Q to quit
            break
        elif key == 32:  # Space to play/pause
            is_playing = not is_playing
        elif key == ord('d') or key == ord('D') or raw_key == 2552445 or raw_key == 65363 or raw_key == 39:  # Right Arrow or D
            current_idx = (current_idx + 1) % len(image_paths)
        elif key == ord('a') or key == ord('A') or raw_key == 2424832 or raw_key == 65361 or raw_key == 37:  # Left Arrow or A
            current_idx = (current_idx - 1) % len(image_paths)
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('4'):
            mode = 4
        elif key == ord('5'):
            mode = 5
        elif key == ord('0') or key == ord('c') or key == ord('C'):
            mode = 0
        elif key == ord('w') or key == ord('W') or key == ord('+') or key == 61:  # Increase speed (decrease delay)
            delay = max(5, delay - 20)
        elif key == ord('s') or key == ord('S') or key == ord('-') or key == 45:  # Decrease speed (increase delay)
            delay = min(2000, delay + 20)
            
        # Automatic advancement when playing
        if is_playing and raw_key == -1:
            current_idx = (current_idx + 1) % len(image_paths)

    cv2.destroyAllWindows()
    print("Player closed.")

if __name__ == '__main__':
    main()
