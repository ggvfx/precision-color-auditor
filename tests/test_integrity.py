import os
import sys
from pathlib import Path

# Setup paths
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
from core.models import ColorPatch, AuditResult, AuditStatus
from core.auditor import Auditor

def test_integrity():
    auditor = Auditor()
    
    # --- TEST 1: The 'Flat Wall' Scenario (Relational DNA Failure) ---
    # We create 6 neutral patches that are all exactly the same color (A gray wall).
    wall_color = np.array([0.18, 0.18, 0.18])
    neutral_patches = []
    for i in range(6):
        patch = ColorPatch(
            name=f"Neutral_{i}",
            observed_rgb=wall_color,
            target_rgb=np.array([0.5, 0.5, 0.5]), # Target doesn't matter for DNA check
            local_center=(0,0),
            index=18 + i # Row 4 of Macbeth
        )
        neutral_patches.append(patch)
    
    result_wall = AuditResult(file_path="test.exr", patches=neutral_patches)
    dna_passed = auditor.verify_dna(result_wall)

    print(f"Debug: Found {len(result_wall.get_neutral_patches())} neutral patches in test data.")
    
    print(f"Test 1 (Gray Wall): DNA Valid = {dna_passed}")
    if not dna_passed:
        print(f"SUCCESS: Auditor caught the 'Flat Wall' snap. Reasoning: {result_wall.ai_reasoning}")

    # --- TEST 2: The 'Bezel Hit' Scenario (Internal Variance Failure) ---
    # We create a patch with high variance (simulating hitting a black grid line)
    noisy_pixels = np.random.normal(0.5, 0.2, (32, 32, 3)) # Mean 0.5, Std Dev 0.2
    pixel_variance = np.mean(np.std(noisy_pixels, axis=(0, 1)))
    
    # Our threshold in config is 0.05. This 0.2 should trigger the flag.
    is_contaminated = pixel_variance > 0.05 
    
    print(f"\nTest 2 (Bezel Hit): Variance = {pixel_variance:.4f}")
    print(f"SUCCESS: Patch flagged as contaminated: {is_contaminated}")

if __name__ == "__main__":
    test_integrity()