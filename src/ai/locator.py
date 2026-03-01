"""
Locator Module - Macro Vision Pass
----------------------------------
Responsibility: Environment-aware object detection.

This module performs the 'Macro' pass of the pipeline. It scans the full, 
high-resolution image to find the physical boundary of a color chart or 
gray card amidst environmental noise (background, hands, lighting fixtures).

Output: A rectified (flat) 1200x800 image buffer and the 4 corner 
coordinates used for the transformation.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from .utils import prep_for_pil
from core.config import settings

class ChartLocator:
    def __init__(self, engine):
        self.engine = engine

    def locate(self, image_buffer: np.ndarray, manual_corners: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the chart and returns (rectified_buffer, corners_array).
        If manual_corners is provided, AI detection is skipped.
        """
        
        # --- BRANCH 1: Manual Override ---
        if manual_corners is not None:
            # Ensure they are the correct shape (4, 2) and float32
            corners = manual_corners.astype(np.float32)
            rectified_image = self.rectify(image_buffer, corners)
            return rectified_image, corners

        # --- BRANCH 2: AI Detection ---
        # 1. Ask the engine to find the chart polygon
        roi_result = self.engine.detect_chart_roi(image_buffer)
        
        # 2. Extract the polygon points
        poly_points = self.engine.get_absolute_bbox(roi_result)
        
        # Check if we actually got points back
        if len(poly_points) == 0:
            print("[WARNING] Locator: Engine returned no polygon points.")
            return None, None
            
        # 3. Use OpenCV to find the 4 tightest corners of the polygon
        # This is what handles the rotation/tilt of the chart
        rect = cv2.minAreaRect(poly_points.astype(np.float32))
        box = cv2.boxPoints(rect) 
        corners = np.array(box, dtype=np.float32)

        # 4. Rectify (Warp)
        rectified_image = self.rectify(image_buffer, corners)
        
        return rectified_image, corners
    
    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Warps the input image to a flat 1200x800 buffer based on 4 corners."""
        # Standard destination coordinates [TL, TR, BR, BL]
        dst_w, dst_h = settings.rectified_size
        dst_points = np.array([
            [0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]
        ], dtype=np.float32)

        # Calculate the transformation matrix
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        
        # Perform the warp
        rectified = cv2.warpPerspective(image_buffer, matrix, (dst_w, dst_h))
        return rectified