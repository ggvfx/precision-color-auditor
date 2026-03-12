import os
import sys
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

import os
import xml.etree.ElementTree as ET
from src.exporters.cdl_writer import CDLWriter
from src.exporters.lut_writer import LUTWriter
from core.models import AuditResult
import numpy as np

def test_cdl_export():
    # Mock a result
    res = AuditResult(
        file_path="C:/Project/Plate.exr",
        template_name="macbeth_24",
        slope=np.array([1.2, 1.1, 0.9], dtype=np.float32),
        offset=np.array([0.01, -0.01, 0.0], dtype=np.float32),
        power=np.array([1.0, 1.0, 1.0], dtype=np.float32), # Added required field
        analysis_intent="neutralize"
    )
    
    out_path = "tests/output/test_export.cdl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    CDLWriter.write(res, out_path)
    
    # Verify file exists
    assert os.path.exists(out_path)
    
    # To find tags in a namespaced XML, we provide the namespace map
    tree = ET.parse(out_path)
    root = tree.getroot()
    ns = {'cdl': 'urn:ASC:CDL:v1.01'}
    
    # Search using the namespace prefix
    slope_node = root.find(".//cdl:Slope", ns)
    
    if slope_node is None:
        # Fallback check: Some parsers might not handle the prefix correctly
        # Let's print the root tag to debug if it fails
        print(f"Debug: Root tag is {root.tag}")
        raise AssertionError("Could not find Slope node in exported CDL.")

    slope_text = slope_node.text
    print(f"Found Slope Text: {slope_text}")
    assert "1.200000" in slope_text
    print("SUCCESS: CDL Exported and Verified.")

def test_lut_export():
    # 1. Setup a result with a known 3x3 Matrix (e.g., swapping R and G)
    res = AuditResult(
        file_path="C:/Project/Plate.exr",
        template_name="macbeth_24",
        slope=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        offset=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        matrix_3x3=np.array([
            [0, 1, 0], # Red input goes to Green output
            [1, 0, 0], # Green input goes to Red output
            [0, 0, 1]  # Blue stays Blue
        ], dtype=np.float32),
        analysis_intent="neutralize"
    )
    
    out_path = "tests/output/test_export.cube"
    
    # 2. Generate a small 2x2x2 LUT for easy verification
    # A 2x2x2 LUT only samples the corners: (0,0,0) through (1,1,1)
    LUTWriter.write_3d_cube(res, out_path, size=2)
    
    assert os.path.exists(out_path)
    
    with open(out_path, "r") as f:
        lines = f.readlines()
        
    # 3. Verify Header
    assert "LUT_3D_SIZE 2" in "".join(lines)
    
    # 4. Verify Data (The "Red" corner should now be Green)
    # The first line after header/comments for a 2x2x2 (R fastest) 
    # usually represents 0 0 0. 
    # The line representing 1 0 0 (Pure Red input) should be 0 1 0 (Pure Green output)
    data_lines = [l for l in lines if l[0].isdigit()]
    
    # In a .cube, (1,0,0) is the second entry if Red is fastest
    # Input: [1, 0, 0] -> Matrix -> Output: [0, 1, 0]
    red_input_result = data_lines[1].strip()
    print(f"Red Corner Result: {red_input_result}")
    
    assert red_input_result == "0.000000 1.000000 0.000000"
    print("SUCCESS: LUT Exported and verified Matrix swap.")

if __name__ == "__main__":
    test_cdl_export()
    test_lut_export()