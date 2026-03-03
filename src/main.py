"""
Application Orchestration & Entry Point.

Links the AI detection layer, color engine, and UI event loop.
Coordinates the end-to-end automated audit and export workflow.
"""

import sys
from pathlib import Path
import cv2

# Project Imports
from src.core.ingest import ImageIngestor
from src.ai.sampler import PatchSampler
from src.ai.engine import ChartDetector
from src.ai.topology import ChartTopology
from src.core.config import settings

def process_audit(image_path: str):
    """
    The Master Workflow for Phase 3.
    """
    print(f"\n[AUDIT START] Target: {Path(image_path).name}")

    # 1. INGEST: Load pixels and extract metadata
    # This is where your new 'generic' and 'raw' logic lives
    pixels, metadata = ImageIngestor.load_image(image_path)
    print(f"[INGEST] Format: {metadata['file_format']} | Space: {metadata['colorspace_hint']}")

    # 2. DETECT & SAMPLE: AI Vision Phase
    # We use the detector to find the chart and the sampler to get patches
    detector = ChartDetector()
    sampler = PatchSampler(detector)
    topology = ChartTopology()
    
    # sampler.sample_all handles the rectification and flip logic internally
    patches, rectified_image, raw_points = sampler.sample_all(pixels, image_path)

    if not raw_points is not None:
        print("[ERROR] Chart could not be located. Aborting.")
        return

    # 3. LINEARIZATION (Phase 3 Current Task)
    # Check if we need to convert sRGB pixels to Linear ACEScg before Auditing
    # This is where OCIO will eventually sit.
    
    # 4. AUDIT: Compare patches to color_targets
    # (Next step: Calculating Delta E and CDLs)
    for patch in patches:
        # Here we will eventually trigger the Auditor
        pass

    # 5. EXPORT: Save the Audit Proof
    export_path = settings.output_dir / f"{Path(image_path).stem}_AUDIT.png"
    cv2.imwrite(str(export_path), cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR))
    print(f"[SUCCESS] Audit Proof saved to: {export_path}")

if __name__ == "__main__":
    # For now, let's test one of your standard assets
    test_file = "D:/_repos/precision-color-auditor/test_assets/macbeth.jpg"
    process_audit(test_file)