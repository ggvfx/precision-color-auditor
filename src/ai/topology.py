"""
Topology Module - Micro Vision Pass
-----------------------------------
Responsibility: Deterministic Grid Mapping.

This module performs the 'Micro' pass. Operating on the rectified crop 
provided by the Locator/Rectifier, it maps a mathematical grid (e.g., 6x4) 
across the image to determine the precise local centroids for each patch.

Output: A list of (y, x) local coordinates representing the 
calculated patch centers on the rectified image.
"""

import numpy as np
import cv2
from core.config import settings
from .utils import prep_for_pil # Keep for PIL-based exports later

class ChartTopology:
    def __init__(self):
        pass

    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warps the skewed camera image into a flat, standardized 1200x800 buffer.
        Detects if the chart is vertical and adjusts the warp to prevent squashing.
        """
        template = settings.get_current_template()
        rect_w, rect_h = template.rectified_size

        # 1. Detect if the bounding box is vertical (Portrait)
        # Using distance between Corner 0 (TL) and Corner 1 (TR) vs Corner 3 (BL)
        dist_w = np.linalg.norm(corners[0] - corners[1])
        dist_h = np.linalg.norm(corners[0] - corners[3])
        is_portrait = dist_h > dist_w

        # 2. Swap destination dimensions if chart is vertical to maintain aspect ratio
        if is_portrait:
            target_w, target_h = rect_h, rect_w
        else:
            target_w, target_h = rect_w, rect_h

        # Define destination points using target variables
        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)

        # 3. Calculate matrix and warp
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        rectified = cv2.warpPerspective(image_buffer, matrix, (target_w, target_h))

        # 4. Canonicalize: If it was portrait, rotate it back to the template's landscape orientation
        if is_portrait:
            rectified = np.rot90(rectified, k=-1) # 90 deg clockwise rotate

        return rectified
    
    def verify_orientation(self, rectified_image: np.ndarray, points: list) -> np.ndarray:
        """
        Uses template-defined anchors to ensure the chart isn't upside down.
        """
        template = settings.get_current_template()
        
        # Only proceed if the template defines an orientation check (e.g., Macbeth)
        if not hasattr(template, 'orientation_anchor') or template.orientation_anchor is None:
            return rectified_image

        idx_bright, idx_dark = template.orientation_anchor
        
        # Sample coordinates from the provided grid/anchor points
        y_b, x_b = points[idx_bright]
        y_d, x_d = points[idx_dark]

        # Check mean luminance (average of RGB) at those coordinates
        val_bright = np.mean(rectified_image[y_b, x_b])
        val_dark = np.mean(rectified_image[y_d, x_d])

        # If the expected bright patch is darker than the dark one, flip 180
        if val_bright < val_dark:
            return np.rot90(rectified_image, 2)
        
        return rectified_image

    def analyze(self) -> list:
        """
        Calculates the patch centers with a safety margin to avoid the thick outer frame.
        """
        template = settings.get_current_template()
        
        # 1. Fetch template-specific dimensions and margin
        rect_w, rect_h = template.rectified_size
        margin = template.inset_margin

        points = []
        
        # 2. Calculate pixel-based margins from the template's percentage
        margin_x = rect_w * margin 
        margin_y = rect_h * margin
        
        # 3. Define the "Working Area" inside the safety margin
        safe_w = rect_w - (2 * margin_x)
        safe_h = rect_h - (2 * margin_y)

        # GRID TOPOLOGY (Macbeth, etc.)
        if template.topology == "grid":
            cols, rows = template.grid
            
            # Divide the SAFE area by the number of patches
            cell_w = safe_w / cols
            cell_h = safe_h / rows

            for r in range(rows):
                for c in range(cols):
                    # Start from the margin, then add the cell center offset
                    center_x = margin_x + (c * cell_w) + (cell_w / 2)
                    center_y = margin_y + (r * cell_h) + (cell_h / 2)
                    points.append((int(center_y), int(center_x)))
        
        # ANCHORED TOPOLOGY (Kodak, etc.)
        elif template.topology == "anchored":
            for anchor_id, data in template.anchors.items():
                u, v = data["pos"]
                center_x = u * rect_w
                center_y = v * rect_h
                points.append((int(center_y), int(center_x)))

        return points

    def generate_qc_image(self, rectified_image: np.ndarray, points: list) -> np.ndarray:
        """
        Overlays sample points onto the rectified image for visual audit.
        """
        # Create a uint8 copy for drawing
        qc_img = (np.clip(rectified_image.copy(), 0, 1) * 255).astype(np.uint8)
        
        for (y, x) in points:
            # Magenta dot with a black border for maximum contrast
            cv2.circle(qc_img, (x, y), 10, (255, 0, 255), -1)
            cv2.circle(qc_img, (x, y), 10, (0, 0, 0), 2)
            
        return qc_img