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
        height, width = image_buffer.shape[:2]
        
        if manual_corners is not None:
            poly_points = manual_corners.astype(np.float32)
        else:
            roi_result = self.engine.detect_chart_roi(image_buffer)
            poly_points = self.engine.extract_polygons(roi_result, width, height)
        
        if poly_points is None or len(poly_points) == 0:
            print("[WARNING] Locator: Engine returned no points.")
            return None, None

        # --- The Snap ---
        refined_points = self._refine_corners(image_buffer, poly_points)
        
        return image_buffer, refined_points

    def _get_intersection(self, line1, line2):
        """Calculates the intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)])

    def _refine_corners(self, image_buffer: np.ndarray, poly_points: np.ndarray) -> np.ndarray:
        """
        Refines AI points by finding dominant lines near the AI edges 
        and calculating their intersections.
        """
        # 1. Standardize to grayscale
        if image_buffer.dtype != np.uint8:
            gray = (np.clip(image_buffer, 0, 1) * 255).astype(np.uint8)
        else:
            gray = image_buffer
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Mask the search area
        mask = np.zeros_like(gray)
        padding = 15 # Reduced slightly for a tighter snap
        cv2.fillPoly(mask, [poly_points.astype(np.int32)], 255)
        mask = cv2.dilate(mask, np.ones((padding, padding), np.uint8))
        
        # 3. Detect Edges and Lines
        edges = cv2.Canny(blurred, 50, 150)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=10)
        
        if lines is None:
            return poly_points

        # 4. Group lines 
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1): h_lines.append(line[0])
            else: v_lines.append(line[0])

        if len(h_lines) < 2 or len(v_lines) < 2:
            return poly_points

        print(f"[DEBUG] Snap active: Found {len(h_lines)}H and {len(v_lines)}V lines.")

        # 5. Proximity Sorting (Select lines closest to AI box boundaries)
        ai_top = np.min(poly_points[:, 1])
        ai_bot = np.max(poly_points[:, 1])
        ai_left = np.min(poly_points[:, 0])
        ai_right = np.max(poly_points[:, 0])

        top_l = min(h_lines, key=lambda l: abs(((l[1] + l[3]) / 2) - ai_top))
        bot_l = min(h_lines, key=lambda l: abs(((l[1] + l[3]) / 2) - ai_bot))
        left_l = min(v_lines, key=lambda l: abs(((l[0] + l[2]) / 2) - ai_left))
        right_l = min(v_lines, key=lambda l: abs(((l[0] + l[2]) / 2) - ai_right))

        # 6. Intersect the 4 boundary lines
        tl = self._get_intersection(top_l, left_l)
        tr = self._get_intersection(top_l, right_l)
        br = self._get_intersection(bot_l, right_l)
        bl = self._get_intersection(bot_l, left_l)

        if any(p is None for p in [tl, tr, br, bl]):
            return poly_points

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def rectify(self, image_buffer: np.ndarray, corners: np.ndarray) -> np.ndarray:
        return image_buffer