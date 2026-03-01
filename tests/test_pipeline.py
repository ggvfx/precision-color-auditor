import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from ai.engine import ChartDetector
from ai.sampler import PatchSampler
from core.config import settings

def mock_load_jpg(path: str):
    """Temporary loader to bypass OIIO requirement for .jpg testing."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img).astype(np.float32) / 255.0

def run_integration_test(image_path: str, label: str):
    print(f"\n--- DIAGNOSTIC RUN: {label} ---")
    
    try:
        # 1. Load Pixels
        pixels = mock_load_jpg(image_path)
        print(f"[SUCCESS] Buffer Loaded: {pixels.shape}")

        # 2. Setup AI
        detector = ChartDetector()
        
        # 3. Diagnostic "Sample" 
        # In this stripped-back version, this only performs detection and saves a QC image
        sampler = PatchSampler(detector)
        patches, full_res_image, raw_points = sampler.sample_all(pixels, image_path)
        
        # 4. Visual Validation Feedback
        if raw_points is not None and len(raw_points) > 0:
            print(f"[SUCCESS] AI found {len(raw_points)} vertices.")
            print(f"[ACTION REQUIRED] Check the overlay image in: {settings.output_dir}")
            print(f"[DEBUG] Raw Coordinates:\n{raw_points}")
        else:
            print("[FAILURE] Vision model could not identify the chart.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAILURE] {e}")

if __name__ == "__main__":
    # Ensure the export directory exists
    settings.output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    #Color charts
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/macbeth.jpg", "Macbeth")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/blackMacbeth.jpeg", "BlackMacbeth")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/pinkMacbeth.jpg", "PinkMacbeth")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/MacbethBalls.jpg", "MacbethBalls")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/miniMacbeth.jpg", "MacbethMini")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/scifiMacbeth.jpg", "MacbethScifi")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/macbeth_ref.jpg", "MacbethRef")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/macbeth_ref2.jpg", "MacbethRefWide")