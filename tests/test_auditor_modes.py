import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import numpy as np
from core.auditor import Auditor
from core.models import AuditResult, ColorPatch
from core.config import settings
from core.templates import KODAK_GRAY_PLUS, MACBETH_24

def test_auditor_strategies():
    auditor = Auditor()
    dummy_pos = (0, 0)

    # --- CASE A: KODAK ANCHORS ---
    settings.active_chart_type = "kodak_gray_plus"
    kodak_result = AuditResult(file_path="simulated_kodak.exr")
    
    # Ensure these names match the 'neutral_indices' in templates.py exactly
    kodak_result.patches = [
        ColorPatch(name="Patch_main_gray", index=0, local_center=dummy_pos,
                   observed_rgb=np.array([0.09, 0.09, 0.09]), 
                   target_rgb=np.array([0.18, 0.18, 0.18])),
        
        ColorPatch(name="Patch_top_left_black", index=1, local_center=dummy_pos,
                   observed_rgb=np.array([0.05, 0.05, 0.05]), 
                   target_rgb=np.array([0.031, 0.031, 0.031])),
        
        ColorPatch(name="Patch_bot_left_white", index=2, local_center=dummy_pos,
                   observed_rgb=np.array([0.43, 0.43, 0.43]), 
                   target_rgb=np.array([0.865, 0.865, 0.865]))
    ]
    
    auditor.perform_audit(kodak_result)
    print(f"\n--- Kodak (Anchors Mode) ---")
    print(f"Calculated {kodak_result.get_sop_summary()}")

    # --- CASE B: MACBETH RAMP ---
    # Simulation: 10% overshoot on all neutral patches
    settings.active_chart_type = "macbeth_24"
    macbeth_result = AuditResult(file_path="simulated_macbeth.exr")
    
    for i in range(18, 24):
        targ_val = MACBETH_24.color_targets[i][0]
        macbeth_result.patches.append(
            ColorPatch(name=f"Patch_{i}", index=i, local_center=dummy_pos,
                       observed_rgb=np.array([targ_val, targ_val, targ_val]) * 1.1,
                       target_rgb=np.array([targ_val, targ_val, targ_val]))
        )
    
    auditor.perform_audit(macbeth_result)
    print(f"\n--- Macbeth (Color/Ramp Mode) ---")
    print(f"Calculated {macbeth_result.get_sop_summary()}")
    print(f"Matrix (Identity Check):\n{macbeth_result.matrix_3x3}")

if __name__ == "__main__":
    test_auditor_strategies()