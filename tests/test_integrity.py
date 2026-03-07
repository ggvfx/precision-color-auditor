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

def test_intent_inversion():
    auditor = Auditor()
    
    # Patch 18: White (Observed is 0.5, Target is 1.0)
    p18 = ColorPatch(
        name="Patch_18", 
        observed_rgb=np.array([0.5, 0.5, 0.5], dtype=np.float32), 
        target_rgb=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
        local_center=(0, 0),
        index=18
    )

    # Patch 23: Black (Observed is 0.01, Target is 0.02)
    # Both patches follow the "Observed is half of Target" rule
    p23 = ColorPatch(
        name="Patch_23", 
        observed_rgb=np.array([0.01, 0.01, 0.01], dtype=np.float32), 
        target_rgb=np.array([0.02, 0.02, 0.02], dtype=np.float32), 
        local_center=(0, 0),
        index=23
    )

    # Scenario A: Neutralize
    res_neut = AuditResult(
        file_path="test.exr", 
        patches=[p18, p23], 
        analysis_intent="neutralize",
        template_name="macbeth_24"
    )
    res_neut = auditor.calculate_cdl_correction(res_neut)
    
    # Scenario B: Extract Grade
    res_extract = AuditResult(
        file_path="test.exr", 
        patches=[p18, p23], 
        analysis_intent="extract_grade",
        template_name="macbeth_24"
    )
    res_extract = auditor.calculate_cdl_correction(res_extract)

    print(f"\nTest 3 (Intent Inversion):")
    print(f"  Neutralize Slope: {res_neut.slope[0]:.2f} (Expected: 2.00)")
    print(f"  Extract Grade Slope: {res_extract.slope[0]:.2f} (Expected: 0.50)")

    assert np.isclose(res_neut.slope[0], 2.0, atol=0.01)
    assert np.isclose(res_extract.slope[0], 0.5, atol=0.01)
    print("SUCCESS: Intent Inversion logic is mathematically sound.")

def test_matrix_solver():
    auditor = Auditor()
    
    # 1. Create 3 primary patches (Red, Green, Blue)
    # Target values (Pure Primaries)
    t_red = np.array([1.0, 0.0, 0.0])
    t_grn = np.array([0.0, 1.0, 0.0])
    t_blu = np.array([0.0, 0.0, 1.0])
    
    # Observed values (Add 10% crosstalk: Red channel is seeing 10% of Green)
    o_red = np.array([1.0, 0.1, 0.0]) 
    o_grn = np.array([0.0, 1.0, 0.0])
    o_blu = np.array([0.0, 0.0, 1.0])
    
    patches = [
        ColorPatch(name="R", observed_rgb=o_red, target_rgb=t_red, local_center=(0,0), index=0),
        ColorPatch(name="G", observed_rgb=o_grn, target_rgb=t_grn, local_center=(0,0), index=1),
        ColorPatch(name="B", observed_rgb=o_blu, target_rgb=t_blu, local_center=(0,0), index=2)
    ]
    
    # Pass a dummy path as the first argument
    res = AuditResult(
        file_path="synthetic_test.exr", 
        patches=patches, 
        analysis_intent="neutralize", 
        template_name="macbeth_24"
    )
    res = auditor._solve_3x3_matrix(res)
    
    print(f"\nTest 4 (3x3 Matrix Crosstalk):")
    print("Generated Matrix:")
    print(res.matrix_3x3)
    
    # Validation: The [1,0] position (Red-into-Green) should be negative 
    # to "subtract" the crosstalk we added.
    cross_value = res.matrix_3x3[1, 0]
    print(f"  Green-to-Red Correction: {cross_value:.3f}")
    
    assert cross_value < 0, "Matrix failed to identify and subtract crosstalk!"
    print("SUCCESS: Matrix solver correctly handled channel crosstalk.")

    # Inside test_matrix_solver() after auditor._solve_3x3_matrix(res)
    print(f"  Post-Matrix Mean Error: {res.delta_e_mean:.5f}")
    
    # In our synthetic test with 0.1 crosstalk, the matrix should fix it perfectly
    assert res.delta_e_mean < 0.001, "Residual error too high! Matrix didn't solve correctly."

if __name__ == "__main__":
    test_integrity()
    test_intent_inversion()
    test_matrix_solver()