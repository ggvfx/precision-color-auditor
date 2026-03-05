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

def test_sampler_memory_flow():
    sampler = PatchSampler(engine=None)
    
    # 1. Create dummy buffers (Simulating a 1920x1080 image)
    # Display: 8-bit (0-255), Audit: Float (Linear)
    display_buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
    audit_buffer = np.zeros((1080, 1920, 3), dtype=np.float32)
    
    # Draw a white square in the middle so the locator has something to find
    cv2.rectangle(display_buffer, (400, 200), (1500, 800), (255, 255, 255), -1)
    
    # Define manual corners so we don't rely on AI detection for this unit test
    manual_corners = np.array([
        [400, 200], [1500, 200], [1500, 800], [400, 800]
    ], dtype=np.float32)

    # 2. Set the global chart type
    settings.active_chart_type = "kodak_gray_plus"
    print(f"Global Setting: {settings.active_chart_type}")

    # 3. Run the sampler
    result = sampler.sample_all(
        display_buffer=display_buffer,
        audit_buffer=audit_buffer,
        source_path="test_image.exr",
        manual_corners=manual_corners
    )

    # 4. VERIFICATION
    print("\n--- Sampler Memory Test Results ---")
    print(f"Pinned Template: {result.template_name}")
    
    if result.rectified_buffer is not None:
        print(f"Rectified Buffer Found: {result.rectified_buffer.shape} type: {result.rectified_buffer.dtype}")
    else:
        print("FAILED: No rectified buffer returned!")

    print(f"Patches Sampled: {len(result.patches)}")
    
    # Test the "Memory" persistence: 
    # Change the global setting and see if the result stays pinned
    settings.active_chart_type = "macbeth_24"
    print(f"Global Setting changed to: {settings.active_chart_type}")
    print(f"Result Object still identifies as: {result.template_name}")

    if result.template_name == "kodak_gray_plus":
        print("SUCCESS: Template pinning is working.")
    else:
        print("FAILED: Result was not pinned to the template.")

if __name__ == "__main__":
    test_sampler_memory_flow()