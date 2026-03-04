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

    def sample_all(self, display_buffer: np.ndarray, audit_buffer: np.ndarray, source_path: str, manual_corners: np.ndarray = None) -> tuple:
        """
        Dual-Branch Pipeline:
        - Locates and Rectifies using display_buffer (Original look).
        - Samples RGB values from audit_buffer (Linear ACEScg).
        """
        template = settings.get_current_template()

        # 1. Locate using the Display Buffer (so AI can see clearly)
        _, raw_points = self.locator.locate(display_buffer, manual_corners=manual_corners)
        
        if raw_points is None or len(raw_points) != 4:
            return [], None, None

        # 2. Rectify BOTH buffers using the same corner points
        # This ensures the coordinates match perfectly between the two branches
        rect_display = self.topology.rectify(display_buffer, raw_points)
        rect_audit = self.topology.rectify(audit_buffer, raw_points)

        # 3. Orientation Fix (Apply to both)
        temp_coords = self.topology.analyze() 
        rect_display = self.topology.verify_orientation(rect_display, temp_coords)
        rect_audit = self.topology.verify_orientation(rect_audit, temp_coords)

        # 4. Generate sample coordinates
        sample_coords = self.topology.analyze()

        color_patches = []
        s_size = template.sample_size
        radius = s_size // 2

        for i, (y, x) in enumerate(sample_coords):
            if template.topology == "anchored":
                target_key = list(template.anchors.keys())[i]
            else:
                target_key = i

            target_rgb = template.color_targets.get(target_key, [0.0, 0.0, 0.0])

            # Define region
            y_start, y_end = max(0, y - radius), min(rect_audit.shape[0], y + radius)
            x_start, x_end = max(0, x - radius), min(rect_audit.shape[1], x + radius)
            
            # 2. Slice all 3 channels (R, G, B) from the audit buffer
            # Since we fixed ColorEngine to reshape(h, w, 3), rect_audit is now 3D.
            patch_roi_audit = rect_audit[int(y_start):int(y_end), int(x_start):int(x_end), :]
            
            # Mean across Height and Width
            mean_color = np.mean(patch_roi_audit, axis=(0, 1))

            patch = ColorPatch(
                name=f"Patch_{target_key}", 
                observed_rgb=mean_color.astype(np.float32),
                target_rgb=np.array(target_rgb, dtype=np.float32),
                local_center=(int(x), int(y)),
                index=i
            )
            color_patches.append(patch)

        # 6. Save Proof using the DISPLAY buffer (Correct original look)
        file_stem = Path(source_path).stem
        qc_image = self.topology.generate_qc_image(rect_display, sample_coords)
        qc_path = settings.output_dir / f"{file_stem}_RECTIFIED.png"
        
        # Ensure we write as uint8 for the display proof
        if qc_image.dtype != np.uint8:
            qc_image = (np.clip(qc_image, 0, 1) * 255).astype(np.uint8)
            
        cv2.imwrite(str(qc_path), cv2.cvtColor(qc_image, cv2.COLOR_RGB2BGR))
        
        print(f"[SUCCESS] Sampled {len(color_patches)} patches. Proof: {file_stem}_RECTIFIED.png")
        
        return color_patches, rect_display, raw_points