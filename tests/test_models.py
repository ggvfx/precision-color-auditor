import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from core.models import ColorPatch, AuditResult, AuditTask
import numpy as np

def test_existing_models_extension():
    # 1. Test your existing ColorPatch validation
    patch = ColorPatch(
        name="Neutral 5",
        observed_rgb=np.array([0.18, 0.18, 0.18], dtype=np.float32),
        target_rgb=np.array([0.18, 0.18, 0.18], dtype=np.float32),
        local_center=(50, 50),
        index=22
    )
    
    # 2. Test your existing AuditResult (aligned to your 'file_path' naming)
    res = AuditResult(file_path="test.exr")
    res.patches.append(patch)
    
    # 3. Test the new AuditTask
    task = AuditTask(task_id="XYZ_001", file_path="test.exr", result=res)
    
    print(f"Task ID: {task.task_id}")
    print(f"Initial Status: {task.status}")
    print(f"Associated File: {task.result.file_path}")
    
    assert task.status == "Pending"
    print("\n[SUCCESS] AuditTask successfully integrated with existing models.")

if __name__ == "__main__":
    test_existing_models_extension()