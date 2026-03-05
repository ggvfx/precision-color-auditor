import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import cv2
import numpy as np
import os
from ai.topology import ChartTopology
from core.models import ColorPatch

def test_visual_states():
    os.makedirs("tests/output", exist_ok=True)
    topo = ChartTopology()
    
    # 1. Create a dummy sRGB chart (Dark grey background)
    img = np.full((600, 900, 3), 50, dtype=np.uint8)
    # Draw a "fake" chart area
    cv2.rectangle(img, (100, 100), (800, 500), (150, 150, 150), -1)
    
    # Define the corners of our fake chart
    corners = np.array([[100, 100], [800, 100], [800, 500], [100, 500]], dtype=np.float32)

    # --- STATE 1: EDIT MODE (User is dragging) ---
    # We only pass the corners, no patch results.
    edit_view = topo.generate_qc_image(img, corners=corners)
    cv2.imwrite("tests/output/qc_state_edit.png", edit_view)
    print("Saved: tests/output/qc_state_edit.png (Boundary Only)")

    # --- STATE 2: AUDIT MODE (Recalculated) ---
    # Wrap RGB lists in np.array to satisfy the Model's dtype check
    mock_patches = [
        ColorPatch(
            name="P1", 
            local_center=(200, 200), 
            delta_e=0.5, 
            observed_rgb=np.array([0.1, 0.1, 0.1], dtype=np.float32), 
            target_rgb=np.array([0.1, 0.1, 0.1], dtype=np.float32), 
            index=0
        ), 
        ColorPatch(
            name="P2", 
            local_center=(400, 200), 
            delta_e=5.0, 
            observed_rgb=np.array([0.5, 0.5, 0.5], dtype=np.float32), 
            target_rgb=np.array([0.1, 0.1, 0.1], dtype=np.float32), 
            index=1
        )
    ]
    
    audit_view = topo.generate_qc_image(img, corners=corners, patch_results=mock_patches)
    cv2.imwrite("tests/output/qc_state_audit.png", audit_view)
    print("Saved: tests/output/qc_state_audit.png (Boundary + Green/Red Dots)")

if __name__ == "__main__":
    test_visual_states()