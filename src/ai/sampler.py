"""
Sampler Module - Signal Extraction & Reconstruction
---------------------------------------------------
Responsibility: Data orchestration and pixel-level sampling.

The Sampler acts as the coordinator for the Locator and Topology modules. 
It manages the data handoff, applies geometric reconstruction (like 
perspective-corrected grids for Macbeth charts), and performs the final 
Center-Weighted RGB extraction from the image buffer.

Output: A list of ColorPatch objects containing the final observed 
RGB signals for the auditor.
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

    def sample_all(self, full_image: np.ndarray, source_name: str, manual_corners: np.ndarray = None) -> tuple:
        """Coordinates the full extraction pipeline."""
        
        # 1. Macro Pass: Get the flat QC image and the corner data
        rectified_crop, corners = self.locator.locate(full_image, manual_corners=manual_corners)
        if rectified_crop is None: 
            print("[WARNING] Locator found no chart.")
            return [], None, None
        
        # 1.5 Save the 'Official QC' crop for user verification
        # Clean the source name to use as a filename
        base_name = Path(source_name).stem
        qc_filename = f"{base_name}_QC_ALIGNMENT.png"
        qc_path = settings.output_dir / qc_filename
        
        # Convert from RGB to BGR for OpenCV saving
        save_buffer = cv2.cvtColor(rectified_crop, cv2.COLOR_RGB2BGR)
        
        # We save as a high-quality PNG to avoid compression artifacts in the QC
        cv2.imwrite(str(qc_path), (np.clip(save_buffer, 0, 1) * 255).astype(np.uint8))
        print(f"[OFFICIAL OUTPUT] Alignment proof saved: {qc_path}")

        # 2. Micro Pass: Get the mathematical grid centers (no AI needed here now)
        points = self.topology.analyze() 
        
        # 3. Final Pixel Sampling
        patches = []
        for i, (py, px) in enumerate(points):
            # Sample using the standardized size from config
            radius = settings.sample_size // 2
            rgb_val = self._get_average_rgb(rectified_crop, py, px, radius)
            
            # Placeholder for target values (these will come from a reference LUT later)
            target_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            patches.append(ColorPatch(
                name=f"Patch_{i}", 
                observed_rgb=rgb_val, 
                target_rgb=target_val,
                local_center=(int(px), int(py)), # New Model Attribute
                index=i
            ))
            
        return patches, rectified_crop, corners
            
    def _get_average_rgb(self, crop: np.ndarray, py: float, px: float, radius: int) -> np.ndarray:
        """
        Samples a small window around a coordinate to avoid noise/hot pixels.
        """
        y, x = int(py), int(px)
        
        # Slicing with bounds safety
        sample_area = crop[
            max(0, y-radius) : min(crop.shape[0], y+radius), 
            max(0, x-radius) : min(crop.shape[1], x+radius)
        ]
        
        # Safety check: if the sample area is empty, return black instead of crashing
        if sample_area.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        return np.mean(sample_area, axis=(0, 1)).astype(np.float32)
