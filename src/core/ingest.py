"""
Precision Color Auditor - Unified Image Ingest Engine
Handles high-precision loading of cinema (EXR, DPX) and RAW formats.
Normalizes all incoming signals to a 32-bit linear float32 buffer.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import rawpy
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Tuple, Dict
from pathlib import Path

class ImageIngestor:
    @staticmethod
    def load_image(file_path: str) -> Tuple[np.ndarray, Dict]:
        ext = os.path.splitext(file_path)[-1].lower()
        raw_extensions = ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.raf', '.rw2', '.ari', '.srw']
        
        try:
            if ext in raw_extensions:
                pixels, meta = ImageIngestor._load_raw(file_path)
            else:
                pixels, meta = ImageIngestor._load_generic(file_path)
            
            # Ensure float32 for high-precision auditing
            if pixels.dtype != np.float32:
                pixels = pixels.astype(np.float32)

            ImageIngestor.validate_signal_range(pixels)
            return pixels, meta
            
        except Exception as e:
            raise IOError(f"Failed to ingest {file_path}: {str(e)}")

    @staticmethod
    def _load_generic(file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Loads non-RAW formats (EXR, DPX, TIFF, JPG) using OpenCV 
        while preserving bit-depth and extracting deep metadata via Pillow.
        """
        # Load pixels with all flags enabled for HDR/High-Bit depth support
        pixels = cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if pixels is None:
            raise IOError(f"Image decoder failed for: {file_path}")
            
        # Standardize BGR -> RGB
        if len(pixels.shape) == 3:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

        # Normalize Integer types to 0.0-1.0 range. 
        # Float types (EXR/HDR) are left as-is for scene-linear data.
        if pixels.dtype == np.uint8:
            pixels = pixels.astype(np.float32) / 255.0
        elif pixels.dtype == np.uint16:
            pixels = pixels.astype(np.float32) / 65535.0

        # Initialize Metadata Container
        metadata = {
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "channels": pixels.shape[2] if len(pixels.shape) > 2 else 1,
            "file_format": Path(file_path).suffix[1:].upper(),
            "is_raw": False,
            "colorspace_hint": "Unknown",
            "raw_metadata": {}
        }

        # Deep Metadata Extraction
        try:
            with Image.open(file_path) as img:
                ext_meta = {}
                for k, v in img.info.items():
                    if isinstance(v, (str, int, float, list)):
                        ext_meta[k] = v
                
                # Fetch exif once
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        
                        # Map standard EXIF to your ground-truth keys
                        if tag_name == 'Make':
                            ext_meta['camera_make'] = value
                        elif tag_name == 'Model':
                            ext_meta['camera_model'] = value
                        else:
                            ext_meta[tag_name] = value
                
                metadata["raw_metadata"] = ext_meta
                
                # Logical Colorspace Guessing
                if "icc_profile" in img.info:
                    metadata["colorspace_hint"] = "ICC Profile Managed"
                elif metadata["file_format"] in ["EXR", "HDR"]:
                    metadata["colorspace_hint"] = "Linear"
                else:
                    metadata["colorspace_hint"] = "sRGB"
        except Exception as meta_error:
            print(f"[METADATA WARNING] Could not parse headers: {meta_error}")

        if metadata["colorspace_hint"] == "Unknown":
            if metadata["file_format"] == "EXR":
                metadata["colorspace_hint"] = "Linear"
            elif metadata["file_format"] in ["JPG", "JPEG", "PNG"]:
                metadata["colorspace_hint"] = "sRGB"

        return pixels, metadata

    @staticmethod
    def _load_raw(file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Processes Digital Camera RAW files into a scene-linear float32 buffer.
        """
        with rawpy.imread(file_path) as raw:
            rgb_linear = raw.postprocess(
                use_camera_wb=True, 
                no_auto_bright=True, 
                output_bps=16, 
                gamma=(1,1)
            )
            pixels = rgb_linear.astype(np.float32) / 65535.0
            
            # Extract values first
            make = getattr(raw, 'camera_make', None)
            model = getattr(raw, 'camera_model', None)

            # Ensure they are strings, not 'bytes'
            if isinstance(make, bytes): make = make.decode('utf-8', errors='ignore')
            if isinstance(model, bytes): model = model.decode('utf-8', errors='ignore')
            
            metadata = {
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "channels": 3,
                "file_format": "Camera RAW",
                "colorspace_hint": "Linear",
                "is_raw": True,
                "raw_metadata": {
                    "camera_make": make,
                    "camera_model": model,
                }
            }
            return pixels, metadata

    @staticmethod
    def validate_signal_range(pixels: np.ndarray):
        """Monitors for HDR peaks or illegal negative values."""
        p_max = np.max(pixels)
        p_min = np.min(pixels)
        if p_max > 1.0:
            print(f"[STATUS] Signal: HDR/Linear detected (Max: {p_max:.3f})")
        if p_min < 0.0:
            print(f"[WARNING] Signal: Negative/Sub-black values (Min: {p_min:.3f})")