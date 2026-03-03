"""
test_audit.py - Full Pipeline Validation
Locator -> Topology -> Sampler -> Auditor
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
from pathlib import Path

import cv2
import numpy as np

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from ai.sampler import PatchSampler
from ai.engine import ChartDetector
from core.auditor import Auditor
from core.config import settings
from core.color_engine import ColorEngine
from core.ingest import ImageIngestor

def run_test_audit(image_path: str):
    # 1. Setup
    # Initialize your AI engine (YOLO/TensorRT/etc.)
    engine = ChartDetector() 
    sampler = PatchSampler(engine)
    auditor = Auditor()
    
    print(f"--- Starting Audit for: {image_path} ---")

    try:
        pixels, metadata = ImageIngestor.load_image(image_path)
    except Exception as e:
        print(f"[ERROR] Ingest failed: {e}")
        return

    # 3. Full Pipeline: Locate, Rectify, and Sample
    # This returns the ColorPatch objects which already have target_rgb inside
    patches, rectified, corners = sampler.sample_all(pixels, image_path)

    if not patches:
        print("[ERROR] Failed to locate or sample the chart.")
        return

    # 4. Perform the Audit
    results = auditor.perform_audit(image_path, patches)

    # 5. Output Results
    print("\n" + "="*50)
    print(f"AUDIT RESULTS: {results.is_pass and 'PASS' or 'FAIL'}")
    print(f"Mean Delta E: {results.delta_e_mean:.3f}")
    print(f"Max Delta E:  {results.delta_e_max:.3f}")
    print("-" * 50)
    print(f"CDL CORRECTION:")
    print(f"  Slope:  {results.slope}")
    print(f"  Offset: {results.offset}")
    print("="*50 + "\n")

    # 6. User Sanity Check: Show the "Audit Proof"
    # The Sampler already saved {filename}_RECTIFIED.png. 
    # Open it to ensure dots are centered.
    print(f"[INFO] Check the output directory for the rectified audit proof.")

if __name__ == "__main__":
    # Test with a known Macbeth chart image
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/linearEXRmacbeth.exr")
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/blackMacbeth.jpeg")
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/blackMacbethrotate.jpeg")
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/macbeth_ref2rotate.jpg")
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/MacbethBalls.jpg")
    run_test_audit("D:/_repos/precision-color-auditor/test_assets/macbeth_ref2rotate.jpg")