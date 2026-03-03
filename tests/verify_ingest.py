import sys
import os
from pathlib import Path

# Path logic
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src"))

# Now these will work
from core.ingest import ImageIngestor

def test_ingest_integrity():
    # 1. Path to a test asset (Update these to your actual paths)
    test_files = [
        "D:/_repos/precision-color-auditor/test_assets/macbeth_ref2.jpg",
        "D:/_repos/precision-color-auditor/test_assets/linearEXRmacbeth.exr",
        "D:/_repos/precision-color-auditor/test_assets/sampleCR2.cr2",
        "D:/_repos/precision-color-auditor/test_assets/sampleRAF.RAF"
    ]

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found: {file_path}")
            continue

        print(f"\n{'='*50}")
        print(f"TESTING INGEST: {os.path.basename(file_path)}")
        print(f"{'='*50}")

        try:
            # RUN THE INGEST
            pixels, metadata = ImageIngestor.load_image(file_path)

            raw_info = metadata.get("raw_metadata", {})
            standard_make = raw_info.get("camera_make", "Unknown")
            standard_model = raw_info.get("camera_model", "Unknown")

            # VERIFY PIXELS
            print(f"SUCCESS: Image loaded as {pixels.dtype}")
            print(f"SHAPE: {pixels.shape}")
            print(f"RANGE: Min {pixels.min():.4f}, Max {pixels.max():.4f}")

            # VERIFY METADATA
            print("\nCORE METADATA:")
            for key in ['width', 'height', 'file_format', 'colorspace_hint', 'is_raw']:
                print(f"  - {key}: {metadata.get(key)}")

            print("\nHEADER/EXIF SAMPLES (raw_metadata):")
            raw_meta = metadata.get("raw_metadata", {})
            # Sample a few common keys to see if Pillow found them
            sample_keys = ['camera_make', 'camera_model', 'ExposureTime', 'ISOSpeedRatings', 'DateTime']
            for sk in sample_keys:
                val = raw_meta.get(sk)
                if val:
                    print(f"  - {sk}: {val}")
                else:
                    print(f"  - {sk}: [Not Found]")

            print("\nALL FOUND METADATA KEYS:")
            print(list(raw_meta.keys()))

        except Exception as e:
            print(f"!!! INGEST FAILED: {str(e)}")

if __name__ == "__main__":
    test_ingest_integrity()