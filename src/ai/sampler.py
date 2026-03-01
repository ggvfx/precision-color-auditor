"""
Sampler Module - Diagnostic Mode
---------------------------------------------------
Responsibility: visual verification of AI detection.
Updated to overlay raw AI points on the source image and skip grid math.
"""

from pathlib import Path
import cv2
import numpy as np

from .locator import ChartLocator
from core.models import ColorPatch
from core.config import settings

class PatchSampler:
    def __init__(self, engine):
        self.locator = ChartLocator(engine)
        # Topology is bypassed for this diagnostic pass
        self.topology = None 

    def sample_all(self, full_image: np.ndarray, source_name: str, manual_corners: np.ndarray = None) -> tuple:
        image_output, raw_points = self.locator.locate(full_image, manual_corners=manual_corners)
        
        if image_output is None or raw_points is None: 
            return [], None, None
        
        h, w = image_output.shape[:2]
        
        # --- DYNAMIC SCALING FOR VISUALS ---
        # Scale markers based on image width (e.g., 1% of width)
        base_size = max(1, w // 100) 
        line_thick = max(2, w // 500)
        font_scale = w / 1500.0 

        draw_buffer = (np.clip(image_output.copy(), 0, 1) * 255).astype(np.uint8)
        draw_buffer = cv2.cvtColor(draw_buffer, cv2.COLOR_RGB2BGR)

        # Draw the points (Scaling 0-1 to pixels)
        # Note: If locator already scaled them, we don't multiply by w,h again
        # Based on your previous logs, raw_points were already absolute pixels
        pixel_points = raw_points if np.max(raw_points) > 1.0 else raw_points * [w, h]

        for i, pt in enumerate(pixel_points):
            x, y = int(pt[0]), int(pt[1])
            # Bright Magenta dots
            cv2.circle(draw_buffer, (x, y), base_size, (255, 0, 255), -1)
            # Large labels
            cv2.putText(draw_buffer, str(i), (x + base_size, y + base_size), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), line_thick)

        # Draw the connection lines
        pts_drawing = pixel_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(draw_buffer, [pts_drawing], isClosed=True, color=(0, 255, 0), thickness=line_thick)

        # Save output
        base_name = Path(source_name).stem
        qc_path = settings.output_dir / f"{base_name}_AI_DETECTION_CHECK.png"
        cv2.imwrite(str(qc_path), draw_buffer)
        
        print(f"[DIAGNOSTIC] AI Detection Proof saved: {qc_path}")
        return [], image_output, pixel_points

    def _get_average_rgb(self, crop, py, px, radius):
        """Not used in Diagnostic Mode"""
        return np.array([0.0, 0.0, 0.0])