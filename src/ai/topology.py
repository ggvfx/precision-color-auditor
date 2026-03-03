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
        template = settings.get_current_template()
        rect_w, rect_h = template.rectified_size

        # 1. Aspect Ratio Analysis
        dist_w = np.linalg.norm(corners[0] - corners[1])
        dist_h = np.linalg.norm(corners[0] - corners[3])
        is_portrait = dist_h > dist_w
        
        print(f"[DEBUG] Geometry: Detected W={dist_w:.1f}, H={dist_h:.1f} | Portrait={is_portrait}")

        if is_portrait:
            target_w, target_h = rect_h, rect_w
        else:
            target_w, target_h = rect_w, rect_h

        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        rectified = cv2.warpPerspective(image_buffer, matrix, (target_w, target_h))

        if is_portrait:
            print("[DEBUG] Rotating Portrait buffer -90deg to Landscape standard.")
            rectified = np.rot90(rectified, k=-1) 

        return rectified

    def verify_orientation(self, rectified_image: np.ndarray, points: list) -> np.ndarray:
        """
        Uses template-defined anchors (18 vs 5) to check for 180-degree flip.
        Uses a luminance ratio to ensure the flip only happens when objectively wrong.
        """
        template = settings.get_current_template()
        
        if not hasattr(template, 'orientation_anchor') or template.orientation_anchor is None:
            return rectified_image

        idx_bright, idx_dark = template.orientation_anchor
        
        # 1. Get coordinates for the diagonal check
        y_b, x_b = points[idx_bright] # Expected White (18)
        y_d, x_d = points[idx_dark]   # Expected Teal (5)

        # 2. Sample 5x5 windows
        box_b = rectified_image[max(0, y_b-2):y_b+3, max(0, x_b-2):x_b+3]
        box_d = rectified_image[max(0, y_d-2):y_d+3, max(0, x_d-2):x_d+3]

        val_bright = np.mean(box_b)
        val_dark = np.mean(box_d)

        # 3. Ratio-based Logic
        # White (18) should be significantly brighter than Teal (5).
        # if Ratio < 1.0, it means the 'Bright' spot is actually the darker patch.
        ratio = val_bright / (val_dark + 1e-6)

        if ratio < 1.0:
            print(f"[DEBUG] Orientation: Flip Triggered (Ratio {ratio:.4f} < 1.0)")
            return np.rot90(rectified_image, 2)
        
        print(f"[DEBUG] Orientation: Correct (Ratio {ratio:.4f} >= 1.0)")
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