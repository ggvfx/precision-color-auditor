import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import os
import shutil
import json
import numpy as np
from pathlib import Path
from core.session import SessionManager
from core.models import AuditResult, ColorPatch

def setup_mock_project(root: Path):
    """Creates a nested VFX-style folder structure."""
    if root.exists():
        shutil.rmtree(root)
    
    # 1. Folders that SHOULD be scanned
    (root / "shot_010/charts").mkdir(parents=True)
    (root / "shot_020/macbeth").mkdir(parents=True)
    
    # 2. Folders that SHOULD be ignored
    (root / "shot_010/plates").mkdir(parents=True)
    (root / "shot_010/renders").mkdir(parents=True)

    # 3. Create Files
    # Valid: in 'charts' folder
    Path(root / "shot_010/charts/chart_v01.exr").touch()
    # Valid: has '_REF' in name
    Path(root / "shot_010/plates/lighting_REF.tif").touch()
    # Invalid: No keyword, wrong folder
    Path(root / "shot_010/plates/actor_01.exr").touch()
    # Valid: in 'macbeth' folder
    Path(root / "shot_020/macbeth/frame_01.jpg").touch()

    # 4. Create a pre-existing Sidecar for one file to test "Recovery"
    sidecar_path = root / "shot_010/charts/chart_v01.exr.pca.json"
    mock_data = {
        "template": "Macbeth_24",
        "is_pass": True,
        "corners": [[0,0], [100,0], [100,100], [0,100]],
        "patches": [{"name": "A1", "delta_e": 0.2, "local_center": [50, 50]}]
    }
    with open(sidecar_path, 'w') as f:
        json.dump(mock_data, f)

def test_session_logic():
    project_root = Path("tests/mock_vfx_project").resolve() 
    setup_mock_project(project_root)
    
    session = SessionManager()

    print("--- Test 1: Recursive Load with Folder Filters ---")
    files = session.load_work_area(
        str(project_root), 
        recursive=True, 
        folder_filters=["charts", "macbeth"]
    )
    print(f"Found {len(files)} files.")
    for f in files:
        print(f"  + {Path(f).relative_to(project_root)}")
    
    assert len(files) == 2, "Should have found exactly 2 files based on folder filters."

    print("\n--- Test 2: Recovery from Sidecar ---")
    # Check if 'chart_v01.exr' has its data restored
    target_path = str((project_root / "shot_010/charts/chart_v01.exr").resolve())
    assert target_path in session.results
    res = session.results[target_path]
    print(f"Restored Template: {res.template_name}")
    print(f"Restored Corners: {res.corners.tolist()}")
    assert res.template_name == "Macbeth_24"

    print("\n--- Test 3: Filename Filtering ---")
    # Clear and reload searching for '_REF' string
    session.results = {}
    ref_files = session.load_work_area(str(project_root), file_filters=["_ref"])
    print(f"Found {len(ref_files)} files with '_REF' in name.")
    assert len(ref_files) == 1
    assert "lighting_REF.tif" in ref_files[0]

    print("\n--- Test 4: Saving a New Sidecar ---")
    # Manually add a result and save it
    new_img = ref_files[0]
    session.results[new_img] = AuditResult(
        file_path=new_img,
        template_name="Test_Template",
        corners=np.array([[1,1],[2,2],[3,3],[4,4]]),
        is_pass=False,
        patches=[],
        ai_reasoning="Manual Test"
    )
    session.save_sidecar(new_img)
    
    expected_json = Path(new_img).with_suffix(Path(new_img).suffix + ".pca.json")
    assert expected_json.exists(), "Sidecar JSON was not created!"
    print(f"Successfully saved sidecar to: {expected_json.name}")

if __name__ == "__main__":
    test_session_logic()