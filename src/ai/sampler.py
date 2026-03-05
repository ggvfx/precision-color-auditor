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
from core.models import ColorPatch, AuditResult  # Added AuditResult
from core.config import settings

class PatchSampler:
    def __init__(self, engine):
        self.engine = engine
        self.locator = ChartLocator(engine)
        self.topology = ChartTopology()

    def sample_all(self, display_buffer: np.ndarray, audit_buffer: np.ndarray, source_path: str, manual_corners: np.ndarray = None) -> AuditResult:
        """
        Dual-Branch Pipeline:
        - Locates and Rectifies using display_buffer.
        - Samples RGB values from audit_buffer.
        - Returns an AuditResult object for UI and Auditor consumption.
        """
        template = settings.get_current_template()

        # 1. Locate using the Display Buffer
        raw_points, reasoning = self.locator.locate(display_buffer, manual_corners=manual_corners)
        
        if raw_points is None or len(raw_points) != 4:
            return AuditResult(
                file_path=source_path, 
                template_name=template.name,
                ai_reasoning=reasoning, # UI will show why it failed
                is_pass=False
            )

        # 2. Rectify BOTH buffers
        rect_display = self.topology.rectify(display_buffer, raw_points)
        rect_audit = self.topology.rectify(audit_buffer, raw_points)

        # 3. Orientation Fix (Apply to both branches)
        temp_coords = self.topology.analyze() 
        flipped_display = self.topology.verify_orientation(rect_display, temp_coords)
        
        # If the memory object changed, a flip was applied
        if flipped_display is not rect_display:
            rect_display = flipped_display
            rect_audit = np.rot90(rect_audit, 2)

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

            # Define region for sampling
            y_start, y_end = max(0, y - radius), min(rect_audit.shape[0], y + radius)
            x_start, x_end = max(0, x - radius), min(rect_audit.shape[1], x + radius)
            
            patch_roi_audit = rect_audit[int(y_start):int(y_end), int(x_start):int(x_end), :]
            mean_color = np.mean(patch_roi_audit, axis=(0, 1))

            patch = ColorPatch(
                name=f"Patch_{target_key}", 
                observed_rgb=mean_color.astype(np.float32),
                target_rgb=np.array(target_rgb, dtype=np.float32),
                local_center=(int(x), int(y)),
                index=i
            )
            color_patches.append(patch)

        # 5. Create the "Audit View" Proof (Uint8 for UI/Display)
        qc_image = self.topology.generate_qc_image(rect_display, sample_coords)
        if qc_image.dtype != np.uint8:
            qc_image = (np.clip(qc_image, 0, 1) * 255).astype(np.uint8)

        # 6. Build the Final AuditResult
        result = AuditResult(
            file_path=source_path,
            template_name=template.name,
            ai_reasoning=reasoning,
            corners=raw_points,
            rectified_buffer=qc_image,
            patches=color_patches
        )
        
        print(f"[SUCCESS] Prepared AuditResult for {Path(source_path).name} with {len(color_patches)} patches.")
        
        return result