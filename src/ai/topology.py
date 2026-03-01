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
from PIL import Image
from .utils import prep_for_pil

from core.config import settings

class ChartTopology:
    def __init__(self):
        """Initialized with settings-based templates."""
        pass

    def analyze(self) -> list:
        """
        Calculates the center (y, x) of each patch based on the active template.
        Works in the coordinate space of settings.rectified_size.
        """
        template = settings.get_current_template()
        cols, rows = template["grid"]
        rect_w, rect_h = settings.rectified_size

        # Calculate cell dimensions
        cell_w = rect_w / cols
        cell_h = rect_h / rows

        points = []

        # Generate centroids row by row (Top to Bottom, Left to Right)
        for r in range(rows):
            for c in range(cols):
                # Calculate center of the current cell
                center_x = (c * cell_w) + (cell_w / 2)
                center_y = (r * cell_h) + (cell_h / 2)
                
                # Format as (y, x) to maintain consistency with existing pipeline
                points.append((int(center_y), int(center_x)))

        print(f"[DEBUG] Topology mapped {len(points)} patches via {cols}x{rows} grid.")
        return points