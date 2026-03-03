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

        # 2. Rectify (Now includes 90-degree Portrait-to-Landscape fix)
        rectified_image = self.topology.rectify(full_image, raw_points)

        # 3. Handle 180-degree flip logic BEFORE generating final coordinates
        # We use a temporary grid to check orientation
        temp_coords = self.topology.analyze() 
        rectified_image = self.topology.verify_orientation(rectified_image, temp_coords)

        # 4. Generate the definitive sample coordinates from the now-finalized image
        sample_coords = self.topology.analyze()

        # 5. Extract average color for each patch
        color_patches = []

        # Get sample size from template, fallback to global if missing
        s_size = template.sample_size
        radius = s_size // 2

        # Identify if we are using indices (grid) or keys (anchored)
        for i, (y, x) in enumerate(sample_coords):
            
            # 1. Determine the correct key for color_targets
            if template.topology == "anchored":
                # Get the string key (e.g., "main_gray") from the anchors dict
                target_key = list(template.anchors.keys())[i]
            else:
                # Use the integer index (0, 1, 2...)
                target_key = i

            # 2. Fetch the target RGB
            target_rgb = template.color_targets.get(target_key, [0.0, 0.0, 0.0])

            # 3. Create the patch
            patch = ColorPatch(
                name=f"Patch_{target_key}", 
                observed_rgb=mean_color.astype(np.float32),
                target_rgb=np.array(target_rgb, dtype=np.float32),
                local_center=(int(x), int(y)),
                index=i
            )
            color_patches.append(patch)

        # 6. Save the 'Audit Proof' (The rectified image with dots)
        file_stem = Path(source_path).stem
        qc_image = self.topology.generate_qc_image(rectified_image, sample_coords)
        qc_path = settings.output_dir / f"{file_stem}_RECTIFIED.png"
        
        cv2.imwrite(str(qc_path), cv2.cvtColor(qc_image, cv2.COLOR_RGB2BGR))
        
        print(f"[SUCCESS] Sampled {len(color_patches)} patches. Proof: {file_stem}_RECTIFIED.png")
        
        return color_patches, rectified_image, raw_points