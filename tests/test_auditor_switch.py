import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from core.models import AuditResult, ColorPatch
from core.auditor import Auditor
from core.config import settings
import numpy as np

def test_template_switch():
    auditor = Auditor()
    
    # 1. Test Macbeth (Should calculate CDL)
    settings.active_chart_type = "macbeth_24"
    res_macbeth = AuditResult(file_path="test.exr", patches=[
        ColorPatch("Patch_18", np.array([0.5, 0.5, 0.5]), np.array([0.6, 0.6, 0.6]), (0,0), 18),
        ColorPatch("Patch_23", np.array([0.02, 0.02, 0.02]), np.array([0.03, 0.03, 0.03]), (0,0), 23)
    ])
    auditor.perform_audit(res_macbeth)
    print(f"Macbeth Slope: {res_macbeth.slope} (Calculated)")

    # 2. Test Gray Card (Should skip CDL - Slope should stay 1.0)
    # Note: Ensure you have a 'gray_card' template in your library or mock it
    settings.active_chart_type = "kodak_gray_plus" # Often treated as anchored/single
    res_gray = AuditResult(file_path="gray.exr", patches=[
        ColorPatch("Patch_main", np.array([0.18, 0.18, 0.18]), np.array([0.18, 0.18, 0.18]), (0,0), 0)
    ])
    auditor.perform_audit(res_gray)
    print(f"Gray Card Slope: {res_gray.slope} (Skipped/Neutral)")

if __name__ == "__main__":
    test_template_switch()