"""
Locator Module - Diagnostic Pass
----------------------------------
Updated to strip out warping and rectangle-fitting.
Returns raw AI detection points on the original image buffer.
"""

import cv2
import numpy as np
from core.config import settings

class ChartLocator:
    def __init__(self, engine):
        self.engine = engine

    def locate(self, image_buffer: np.ndarray, manual_corners: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagnostic Version: Returns the ORIGINAL image and RAW points.
        Bypasses warping and rectification logic.
        """
        # 1. Get dimensions from the input image (Crucial fix)
        height, width = image_buffer.shape[:2]
        
        # --- BRANCH 1: Manual Override ---
        if manual_corners is not None:
            return image_buffer, manual_corners.astype(np.float32)

        # --- BRANCH 2: AI Detection ---
        # 2. Ask the engine to find the chart polygon
        roi_result = self.engine.detect_chart_roi(image_buffer)
        
        # 3. Extract raw absolute pixel coordinates (Passing width/height)
        poly_points = self.engine.extract_polygons(roi_result, width, height)
        
        if poly_points is None or len(poly_points) == 0:
            print("[WARNING] Locator: Engine returned no points.")
            return None, None

        # --- DIAGNOSTIC STRIP-BACK ---
        # We return the original image buffer and the raw scaled points.
        return image_buffer, poly_points
    
    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        DEPRECATED for Diagnostic Pass. 
        Kept as a stub to avoid breaking sampler imports.
        """
        return image_buffer