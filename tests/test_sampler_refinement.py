import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
import cv2
from ai.sampler import PatchSampler
from core.config import settings

def test_sampler_refinement_logic():
    # 1. Init Sampler (Passing None for engine to test Manual/Snap bypass)
    sampler = PatchSampler(engine=None)
    
    # 2. Setup dummy image with a clear white square (for snapping)
    display_buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(display_buffer, (400, 200), (1500, 800), (255, 255, 255), -1)
    
    # Define corners that are slightly OFF the square (e.g., 5 pixels out)
    off_corners = np.array([
        [395, 195], [1505, 195], [1505, 805], [395, 805]
    ], dtype=np.float32)

    # --- SCENARIO A: SNAP ENABLED ---
    # Result should snap TO the white square (400, 200)
    res_snap = sampler.sample_all(display_buffer, display_buffer.astype(np.float32), 
                                 "test.exr", manual_corners=off_corners, use_snap=True)
    
    print(f"Snap Enabled Corner: {res_snap.corners[0]}")
    # Logic check: It should be closer to 400,200 than the off_corners were.

    # --- SCENARIO B: SNAP DISABLED ---
    # Result must match off_corners EXACTLY (395, 195)
    res_no_snap = sampler.sample_all(display_buffer, display_buffer.astype(np.float32), 
                                    "test.exr", manual_corners=off_corners, use_snap=False)
    
    print(f"Snap Disabled Corner: {res_no_snap.corners[0]}")
    
    assert np.allclose(res_no_snap.corners, off_corners), "FAILED: Snap disabled but points moved!"
    print("SUCCESS: Snap bypass verified.")

if __name__ == "__main__":
    test_sampler_refinement_logic()