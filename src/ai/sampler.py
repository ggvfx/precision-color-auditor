"""
Sampler Module
---------------------------------------------------
Responsibility: High-precision sampling of color patches from rectified chart buffers
"""

from pathlib import Path
import cv2
import numpy as np

from .locator import ChartLocator
from .topology import ChartTopology
from core.models import ColorPatch
from core.config import settings

class PatchSampler:
    def __init__(self, engine):
        self.locator = ChartLocator(engine)
        self.topology = ChartTopology()

    def sample_all(self, full_image: np.ndarray, source_path: str, manual_corners: np.ndarray = None) -> tuple:
        """
        Full Pipeline: Locate -> Rectify -> Map -> Sample.
        Returns: (List of ColorPatches, Rectified Image, Refined Corners)
        """
        template = settings.get_current_template()

        # 1. Locate the 4 corners
        _, raw_points = self.locator.locate(full_image, manual_corners=manual_corners)
        
        if raw_points is None or len(raw_points) != 4:
            return [], None, None

        # 2. Rectify to the standardized 1200x800 flat buffer
        rectified_image = self.topology.rectify(full_image, raw_points)

        # 3. Get the 24 sample coordinates (with that 3% inset we set)
        sample_coords = self.topology.analyze()

        # 4. Extract average color for each patch
        color_patches = []

        # Get sample size from template, fallback to global if missing
        s_size = template.get("sample_size", settings.sample_size)
        radius = s_size // 2

        for i, (y, x) in enumerate(sample_coords):
            # Calculate the window bounds
            y1, y2 = max(0, y - radius), min(rectified_image.shape[0], y + radius)
            x1, x2 = max(0, x - radius), min(rectified_image.shape[1], x + radius)
            
            # Extract the 32x32 window and calculate the mean
            patch_window = rectified_image[int(y1):int(y2), int(x1):int(x2)]
            mean_color = np.mean(patch_window, axis=(0, 1))
            
            # --- THE FIX: Match the ColorPatch Model requirements ---
            # We must provide: name, observed_rgb, target_rgb, local_center, index
            patch = ColorPatch(
                name=f"Patch_{i}", 
                observed_rgb=mean_color.astype(np.float32),
                target_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float32), # Placeholder
                local_center=(int(x), int(y)),
                index=i
            )
            color_patches.append(patch)

        # 5. Save the 'Audit Proof' (The rectified image with dots)
        file_stem = Path(source_path).stem
        qc_image = self.topology.generate_qc_image(rectified_image, sample_coords)
        qc_path = settings.output_dir / f"{file_stem}_RECTIFIED.png"
        
        cv2.imwrite(str(qc_path), cv2.cvtColor(qc_image, cv2.COLOR_RGB2BGR))
        
        print(f"[SUCCESS] Sampled {len(color_patches)} patches. Proof: {file_stem}_RECTIFIED.png")
        
        return color_patches, rectified_image, raw_points