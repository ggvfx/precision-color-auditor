"""
Locator Module - Diagnostic Pass
----------------------------------
Updated to strip out warping and rectangle-fitting.
Returns raw AI detection points on the original image buffer.
"""

import numpy as np
from core.config import settings

class ChartLocator:
    def __init__(self, engine):
        self.engine = engine

    def locate(self, image_buffer: np.ndarray, manual_corners: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Locates the Macbeth chart using either AI detection or manual override.
        """
        # 1. Get dimensions from the input image
        height, width = image_buffer.shape[:2]
        
        # --- BRANCH 1: Manual Override ---
        # If the UI sends coordinates, use them immediately and bypass AI.
        if manual_corners is not None:
            return image_buffer, manual_corners.astype(np.float32)

        # --- BRANCH 2: AI Detection ---
        # Otherwise, proceed with the standard AI grounding flow.
        roi_result = self.engine.detect_chart_roi(image_buffer)
        
        # Extract raw absolute pixel coordinates
        poly_points = self.engine.extract_polygons(roi_result, width, height)
        
        if poly_points is None or len(poly_points) == 0:
            print("[WARNING] Locator: Engine returned no points.")
            return None, None

        # Return the buffer and the initial points (AI or Manual)
        return image_buffer, poly_points
    
    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """ Stub for future rectification logic """
        return image_buffer
    
    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        DEPRECATED for Diagnostic Pass. 
        Kept as a stub to avoid breaking sampler imports.
        """
        return image_buffer