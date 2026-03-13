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
from core.models import ColorPatch, AuditResult, AuditStatus
from core.config import settings
from ai.utils import prep_for_pil
from core.color_engine import ColorEngine

class PatchSampler:
    def __init__(self, color_engine, engine):
        self.color_engine = color_engine
        self.engine = engine
        self.locator = ChartLocator(engine)
        self.topology = ChartTopology()

    def sample_all(self, display_buffer: np.ndarray, audit_buffer: np.ndarray, 
                   source_path: str, manual_corners: np.ndarray = None, 
                   use_snap: bool = True) -> AuditResult:
        """
        Processes the image and returns a fully populated AuditResult.
        """
        template = settings.get_current_template()

        # --- TEMPORARY TEST SABOTAGE ---
        #test_display = display_buffer * 0.01  # Make the UI view nearly pitch black
        #print("[TEST] Sabotaging display_buffer to simulate a dark LUT...")

        # 1. Locate using the Display Buffer
        raw_points, reasoning = self.locator.locate(display_buffer, audit_buffer, manual_corners=manual_corners, use_snap=use_snap)
        
        # Handle Failure Case
        if raw_points is None or len(raw_points) != 4:
            return AuditResult(
                file_path=source_path, 
                template_name=template.name,
                status=AuditStatus.FAILED, # Explicit Status
                ai_reasoning=reasoning,
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

        # 4. Generate sample coordinates & Sample Patches
        sample_coords = self.topology.analyze()
        color_patches = []
        radius = template.sample_size // 2

        total_contamination_flags = 0

        for i, (y, x) in enumerate(sample_coords):
            target_key = list(template.anchors.keys())[i] if template.topology == "anchored" else i
            target_rgb = template.color_targets.get(target_key, [0.0, 0.0, 0.0])

            # ROI Sampling
            y_s, y_e = max(0, int(y - radius)), min(rect_audit.shape[0], int(y + radius))
            x_s, x_e = max(0, int(x - radius)), min(rect_audit.shape[1], int(x + radius))
            
            roi = rect_audit[y_s:y_e, x_s:x_e, :]
            
            # --- INTEGRITY CHECK: INTERNAL VARIANCE ---
            # np.std tells us if the pixels are consistent. High std = noise or bezel.
            pixel_variance = np.mean(np.std(roi, axis=(0, 1)))
            
            # --- INTEGRITY CHECK: CORE VS RING ---
            # Compare inner 50% to full sample. If they differ, we're hitting an edge.
            h, w, _ = roi.shape
            core_roi = roi[h//4:-h//4, w//4:-w//4, :]
            core_mean = np.mean(core_roi, axis=(0, 1))
            full_mean = np.mean(roi, axis=(0, 1))
            
            edge_drift = np.mean(np.abs(core_mean - full_mean))
            
            # Contamination logic
            is_bad = (pixel_variance > settings.integrity_threshold) or (edge_drift > 0.02)
            if is_bad: total_contamination_flags += 1

            patch = ColorPatch(
                name=f"Patch_{target_key}", 
                observed_rgb=full_mean.astype(np.float32),
                target_rgb=np.array(target_rgb, dtype=np.float32),
                local_center=(int(x), int(y)),
                index=i,
                sample_variance=pixel_variance,
                is_contaminated=is_bad
            )
            color_patches.append(patch)

        # 5. Create the "Audit View" Proof (Optimized)
        qc_image_raw = self.topology.generate_qc_image(rect_display, sample_coords)
        # Use our new util to get a clean uint8 buffer for the UI
        qc_image_uint8 = np.array(prep_for_pil(qc_image_raw))

        # 6. Build and Return the Final AuditResult
        # If manual_corners exists, it's a MANUAL_EDIT. Otherwise, it's COMPLETE.
        final_status = AuditStatus.MANUAL_EDIT if manual_corners is not None else AuditStatus.COMPLETE
        
        # Calculate final integrity score for the result
        integrity_score = 1.0 - (total_contamination_flags / len(color_patches))

        return AuditResult(
            file_path=source_path,
            template_name=template.name,
            status=final_status, # Preserved your status logic
            corners=raw_points,
            rectified_buffer=qc_image_uint8,
            patches=color_patches,
            ai_reasoning=reasoning,
            alignment_integrity=integrity_score,
            integrity_warning=integrity_score < 1.0,
            analysis_intent=settings.analysis_intent, 
            is_pass=True # Auditor will finalize this based on Delta E and DNA
        )