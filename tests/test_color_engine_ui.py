import os
import sys
from pathlib import Path

# Setup paths
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

from core.color_engine import ColorEngine
from core.config import settings

def test_ui_lists():
    print("--- OCIO UI List Test ---")
    engine = ColorEngine()
    
    # 1. Get the lists
    source_list, audit_list = engine.get_ui_lists()
    
    # 2. Verify Source List
    print(f"\nTotal Source Spaces found: {len(source_list)}")
    # Print a few examples to see if they look like standard OCIO names
    print(f"Sample Source Spaces: {source_list[:5]}")

    # 3. Verify Audit List (The Filtered List)
    print(f"\nTotal Audit (Linear) Spaces found: {len(audit_list)}")
    
    # Check if ACEScg is at the top as intended
    if audit_list and audit_list[0] == "ACEScg":
        print("SUCCESS: ACEScg is prioritized at index 0.")
    else:
        print("NOTICE: ACEScg is not the primary space or was not found.")

    # Check for unwanted non-linear spaces
    illegal_finds = [s for s in audit_list if "srgb" in s.lower() and "linear" not in s.lower()]
    if illegal_finds:
        print(f"WARNING: Found potential non-linear spaces in Audit list: {illegal_finds}")
    else:
        print("SUCCESS: No obvious non-linear spaces found in Audit list.")

    print("\nFull Audit List for Review:")
    for space in audit_list:
        print(f" - {space}")

if __name__ == "__main__":
    test_ui_lists()