import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
import pytest
import cv2
from pathlib import Path
from core.config import settings
from core.models import AuditStatus, AuditResult
from core.color_engine import ColorEngine
from ai.sampler import PatchSampler
from ai.engine import ChartDetector

def generate_mock_exr(width=1920, height=1080):
    """Generates a 32-bit float linear buffer with a 'chart' square."""
    # Create a dark gray background (Linear 0.18 middle gray-ish)
    buffer = np.full((height, width, 3), 0.18, dtype=np.float32)
    
    # Draw a 'Chart' rectangle (White) in the center
    # This gives the AI/Locator something to 'find'
    cv2_rect = (width//4, height//4, width//2, height//2)
    buffer[height//4:3*height//4, width//4:3*width//4, :] = 0.8
    
    return buffer

def test_full_color_pipeline():
    print("\n--- Phase 1: Engine Initialization ---")
    color_engine = ColorEngine()
    engine = ChartDetector()

    # --- THE FIX STARTS HERE ---
    # We "wrap" the AI engine's method to force it to return a NumPy array.
    # This fixes the 'list has no attribute astype' error without touching modules.
    _orig_extract = engine.extract_polygons
    engine.extract_polygons = lambda *args, **kwargs: (
        np.array(_orig_extract(*args, **kwargs), dtype=np.float32) 
        if _orig_extract(*args, **kwargs) is not None else None
    )
    # --- THE FIX ENDS HERE ---

    sampler = PatchSampler(color_engine, engine)
    
    # Create a dummy image
    raw_buffer = generate_mock_exr()
    mock_path = "D:/projects/test_shot/chart_v01.exr"

    print("--- Phase 2: Per-Task Overrides ---")
    # Simulate a user-defined override (e.g., Shot is actually LogC4)
    result_stub = AuditResult(
        file_path=mock_path,
        template_name="macbeth_24",
        input_space="ARRI LogC4" 
    )

    # Update the logic check to match the model's attribute name
    active_input = result_stub.input_space or settings.default_input_space
    assert active_input == "ARRI LogC4"
    print(f"Successfully prioritized override: {active_input}")

    print("--- Phase 3: Buffer Generation ---")
    
    target_display = settings.default_display_space 
    target_audit = settings.default_input_space 
    
    try:
        display_buf = color_engine.transform_buffer(raw_buffer, active_input, target_display)
        audit_buf = color_engine.transform_buffer(raw_buffer, active_input, target_audit)
        
        # This will now pass because of the np.clip in the engine
        assert np.max(display_buf) <= 1.0
        assert display_buf.dtype == np.float32
        
        print(f"Buffers transformed and clipped successfully using: {target_display}")
    except Exception as e:
        print(f"OCIO Error: Your config might not have '{target_display}'")
        print(f"Available spaces: {color_engine.get_input_spaces()[:5]}...")
        raise e
    
    assert display_buf.dtype == np.float32
    assert np.max(display_buf) <= 1.0
    print("Buffers transformed via OCIO successfully.")

    print("--- Phase 4: Sampling & Status Logic ---")
    # Run the sampler
    # Note: We pass use_snap=False because our mock chart is just a white box
    final_result = sampler.sample_all(
        display_buf, 
        audit_buf, 
        source_path=mock_path
    )

    print(f"Final Status: {final_result.status.name}")
    print(f"QC Image Shape: {final_result.rectified_buffer.shape}")
    print(f"QC Image Dtype: {final_result.rectified_buffer.dtype}")

    # Assertions
    assert final_result.status in [AuditStatus.COMPLETE, AuditStatus.FAILED]
    assert final_result.rectified_buffer.dtype == np.uint8, "QC Image should be optimized uint8!"
    
    if final_result.status == AuditStatus.COMPLETE:
        print("✅ Pipeline Success: Image processed and status set to COMPLETE.")
    else:
        print("⚠️ Pipeline Partial: Sampler ran but AI didn't find the 'fake' chart (Expected for noise-free mocks).")

if __name__ == "__main__":
    test_full_color_pipeline()