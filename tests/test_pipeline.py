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
    print(f"\nTESTING: {label}")
    
    try:
        # 1. Load Pixels
        pixels = mock_load_jpg(image_path)
        print(f"[SUCCESS] Buffer Loaded: {pixels.shape}")

        # 2. Setup AI
        # We still need the base detector (Florence-2) to pass into our modules
        detector = ChartDetector()
        
        # 3. Sample (The New Multi-Pass Way)
        # Sampler now handles Locator -> Topology -> Grid Math internally
        sampler = PatchSampler(detector)
        patches, crop, corners = sampler.sample_all(pixels, image_path)
        
        if crop is not None:
            print(f"[SUCCESS] Discovered {len(patches)} patches.")
            print(f"[QC] Alignment proof should be in: {settings.output_dir}")
        else:
            print("[FAILURE] Chart not found in image.")

        # 4. Signature Match
        sig = settings.get_signature(len(patches))
        if sig:
            print(f"[MATCH] Identified as: {sig['label']}")
        else:
            print(f"[UNRECOGNIZED] Discovered {len(patches)} patches. Layout check required.")

    except Exception as e:
        import traceback
        traceback.print_exc() # Better for debugging these new modules
        print(f"[FAILURE] {e}")

if __name__ == "__main__":
    #Color charts
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/macbeth.jpg", "Macbeth")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/blackMacbeth.jpeg", "BlackMacbeth")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/pinkMacbeth.jpg", "PinkMacbeth")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/KodakColor.jpg", "KodakColor")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/MacbethBalls.jpg", "MacbethBalls")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/miniMacbeth.jpg", "MacbethMini")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/scifiMacbeth.jpg", "MacbethScifi")

    #Grey cards
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/KodakGray.jpg", "Kodak Gray Card Plus")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/greycardboy.jpg", "Gray Card Boy")
    #run_integration_test("D:/_repos/precision-color-auditor/test_assets/Greycardphoto.jpg", "Gray Card Photo")