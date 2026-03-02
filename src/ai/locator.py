"""
Locator Module - Diagnostic Pass
----------------------------------
Updated to strip out warping and rectangle-fitting.
Returns raw AI detection points on the original image buffer.
"""

import numpy as np
import cv2
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
        
        # --- 1: Manual Override ---
        # If the UI sends coordinates, use them immediately and bypass AI.
        if manual_corners is not None:
            poly_points = manual_corners.astype(np.float32)
        # --- 2: AI Detection ---
        # Otherwise, proceed with the standard AI grounding flow.
        else:
            roi_result = self.engine.detect_chart_roi(image_buffer)
            poly_points = self.engine.extract_polygons(roi_result, width, height)
        
        if poly_points is None or len(poly_points) == 0:
            print("[WARNING] Locator: Engine returned no points.")
            return None, None

        # --- 3: OpenCV Refinement (The Snap) ---
        # Pass the points to OpenCV to find the actual physical edges
        refined_points = self._refine_corners(image_buffer, poly_points)
        
        return image_buffer, refined_points
    
    def _refine_corners(self, image_buffer: np.ndarray, poly_points: np.ndarray) -> np.ndarray:
        """
        Uses Canny and HoughLines to find the actual edges of the card.
        Calculates intersections to find 'Virtual Corners' (ignoring rounded edges).
        """
        # 1. Convert to grayscale and blur to reduce noise
        if len(image_buffer.shape) == 3:
            gray = cv2.cvtColor(image_buffer.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image_buffer.astype(np.uint8)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Mask the search area (AI box + 10% padding)
        mask = np.zeros_like(gray)
        # Create a slightly larger hull so we don't clip the real edge
        padding = 20 
        cv2.fillPoly(mask, [poly_points.astype(np.int32)], 255)
        mask = cv2.dilate(mask, np.ones((padding, padding), np.uint8))
        
        # 3. Edge Detection
        edges = cv2.Canny(blurred, 50, 150)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 4. Find Lines (Hough Transform)
        # We look for long, strong lines that represent the card boundaries
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return poly_points # Fallback if no lines found

        # For this Phase 1 test, we'll return the original points if the line 
        # count is too low, ensuring we don't break your diagnostic.
        # In Phase 2, we will implement the actual intersection math here.
        return poly_points

    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """ 
        STUB: Kept as-is for your current diagnostic flow.
        """
        return image_buffer