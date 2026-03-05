import os
import sys
from pathlib import Path

# Setup paths
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
from ai.engine import ChartDetector
from ai.sampler import PatchSampler
import cv2

def test_full_ai_stack():
    print("Initializing Florence-2 (this may take a moment)...")
    engine = ChartDetector()
    sampler = PatchSampler(engine=engine)

    # Load a real image or a very convincing synthetic one
    # For now, we'll create a blank image which should trigger a "Failure" 
    # reasoning string from the AI.
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    print("Running Sampler on blank image to test failure reasoning...")
    result = sampler.sample_all(dummy_img, dummy_img.astype(np.float32), "no_chart.jpg")

    print("\n--- AI REASONING TEST ---")
    print(f"File: {result.file_path}")
    print(f"Success: {not result.is_pass == False}")
    print(f"Raw AI Reasoning: {result.ai_reasoning}")

    if len(result.ai_reasoning) > 0:
        print("SUCCESS: Sampler successfully captured the AI's thoughts.")
    else:
        print("FAILED: Reasoning string is empty.")

if __name__ == "__main__":
    test_full_ai_stack()