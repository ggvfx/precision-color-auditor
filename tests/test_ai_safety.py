import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
import torch
from ai.engine import ChartDetector
from ai.locator import ChartLocator
from ai.sampler import PatchSampler
from core.color_engine import ColorEngine

def run_safety_test():
    print("--- STARTING AI SAFETY INTEGRITY TEST ---")
    
    # 1. Setup our "Brain" and "Eyes"
    engine = ChartDetector()
    sampler = PatchSampler(None, engine) # We pass None for color_engine for this test
    
    # 2. Create a "Fake" Image (1920x1080)
    # This simulates a real image buffer
    fake_audit = np.random.rand(1080, 1920, 3).astype(np.float32)
    
    # 3. Simulate the SABOTAGE
    # We make the display buffer pitch black to force the fallback
    fake_display = np.zeros((1080, 1920, 3), dtype=np.float32)
    
    print("[STEP 1] Attempting detection on pitch-black display buffer...")
    
    # 4. Run the Locator directly to see the handshake
    # This calls the detect_with_fallback method we just wrote
    points, reasoning = sampler.locator.locate(
        fake_display, 
        fake_audit, 
        use_snap=False
    )
    
    print(f"\n[FINAL RESULT] Reasoning: {reasoning}")
    if "FALLBACK-MODE" in reasoning:
        print("✅ SUCCESS: The Safety Gamma fallback was triggered!")
    else:
        print("❌ FAILURE: The system didn't use the fallback.")

if __name__ == "__main__":
    run_safety_test()