import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Path logic (same as before)
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

# Import only the AI modules (skipping ingest.py for this test)
from ai.detector import ChartDetector
from ai.sampler import PatchSampler
from core.config import settings

def mock_load_jpg(path: str):
    """Temporary loader to bypass OIIO requirement for .jpg testing."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img).astype(np.float32) / 255.0

def run_integration_test(image_path: str, label: str):
    print(f"\nTESTING: {label}")
    
    try:
        # 1. Mock Ingest (Using Pillow for .jpg test assets)
        pixels = mock_load_jpg(image_path)
        print(f"[SUCCESS] Buffer Loaded: {pixels.shape}")

        # 2. Detect (AI - Florence-2)
        detector = ChartDetector()
        roi = detector.detect_chart_roi(pixels)
        print(f"[SUCCESS] ROI Found")

        # 3. Sample (AI - Discovery)
        sampler = PatchSampler(detector)
        patches = sampler.discover_and_sample(pixels, roi)
        
        print(f"[SUCCESS] Discovered {len(patches)} patches.")

        # Signature Match
        sig = settings.get_signature(len(patches))
        if sig:
            print(f"[MATCH] Identified as: {sig['label']}")
        else:
            print(f"[UNRECOGNIZED] Discovered {len(patches)} patches. Layout check required.")

    except Exception as e:
        print(f"[FAILURE] {e}")

if __name__ == "__main__":
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/macbeth.jpg", "Macbeth")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/KodakGray.jpg", "Kodak Gray Card Plus")