import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from ai.engine import ChartDetector
from ai.sampler import PatchSampler
from core.config import settings
from ai.topology import ChartTopology

def mock_load_jpg(path: str):
    """Temporary loader to bypass OIIO requirement for .jpg testing."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img).astype(np.float32) / 255.0

def run_integration_test(image_path: str, label: str):
    print(f"\n--- DIAGNOSTIC RUN: {label} ---")

    file_stem = Path(image_path).stem
    
    try:
        pixels = mock_load_jpg(image_path)
        print(f"[SUCCESS] Buffer Loaded: {pixels.shape}")

        detector = ChartDetector()
        topology = ChartTopology() # Initialize Topology
        
        sampler = PatchSampler(detector)
        # Note: raw_points are the 4 refined corners from locator
        patches, full_res_image, raw_points = sampler.sample_all(pixels, image_path)
        
        if raw_points is not None and len(raw_points) == 4:
            print(f"[SUCCESS] AI found 4 vertices.")

            # --- NEW RECTIFICATION STEP ---
            # 1. Warp the chart to the flat 1200x800 buffer
            rectified = topology.rectify(pixels, raw_points)
            
            # 2. Map the sample points (dots)
            sample_points = topology.analyze()
            
            # 3. Create the QC visual
            qc_image = topology.generate_qc_image(rectified, sample_points)
            
            # 4. Save the "Official" Audit Proof
            export_name = f"{file_stem}_RECTIFIED.png"
            export_path = settings.output_dir / export_name
            
            # Convert RGB to BGR for OpenCV saving
            cv2.imwrite(str(export_path), cv2.cvtColor(qc_image, cv2.COLOR_RGB2BGR))
            
            print(f"[DIAGNOSTIC] Rectified Proof saved: {export_path}")
            # ------------------------------

        else:
            print("[FAILURE] Vision model could not identify the 4 corners.")

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

    #GreyCards
    """ run_integration_test("D:/_repos/precision-color-auditor/test_assets/KodakGray.jpg", "KodakGrayCardPlus")
    run_integration_test("D:/_repos/precision-color-auditor/test_assets/KodakGrayCrop.jpg", "KodakGrayCardPlusCrop") """