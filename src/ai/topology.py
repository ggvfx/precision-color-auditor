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
        self.rect_w, self.rect_h = settings.rectified_size

    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Warps the skewed camera image into a flat, standardized 1200x800 buffer.
        """
        # Define destination points for the flat crop
        dst_pts = np.array([
            [0, 0],
            [self.rect_w - 1, 0],
            [self.rect_w - 1, self.rect_h - 1],
            [0, self.rect_h - 1]
        ], dtype=np.float32)

        # Calculate the perspective matrix from 4 refined points
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        
        # Warp the high-precision float buffer
        rectified = cv2.warpPerspective(image_buffer, matrix, (self.rect_w, self.rect_h))
        return rectified

    def analyze(self) -> list:
        """
        Calculates the patch centers with a safety margin to avoid the thick outer frame.
        """
        template = settings.get_current_template()
        points = []
        
        # 2% Inset to account for thicker outer borders
        margin_x = self.rect_w * 0.03 
        margin_y = self.rect_h * 0.03
        
        # The "Working Area" inside the black bezel
        safe_w = self.rect_w - (2 * margin_x)
        safe_h = self.rect_h - (2 * margin_y)

        if template["topology"] == "grid":
            cols, rows = template["grid"]
            
            # Divide the SAFE area by the number of patches
            cell_w = safe_w / cols
            cell_h = safe_h / rows

            for r in range(rows):
                for c in range(cols):
                    # Start from the margin, then add the cell center offset
                    center_x = margin_x + (c * cell_w) + (cell_w / 2)
                    center_y = margin_y + (r * cell_h) + (cell_h / 2)
                    points.append((int(center_y), int(center_x)))
        
        elif template["topology"] == "anchored":
            # Anchored templates (like Kodak) usually define their own safe spots
            for anchor_id, data in template["anchors"].items():
                u, v = data["pos"]
                center_x = u * self.rect_w
                center_y = v * self.rect_h
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