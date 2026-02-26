"""
Precision Color Auditor - Unified Image Ingestion Layer
Handles high-precision loading of cinema and RAW formats using OpenImageIO and Rawpy.
Normalizes all incoming signals to 32-bit float arrays for deterministic auditing.
"""

import os
import numpy as np
import OpenImageIO as oiio
import rawpy
from typing import Tuple, Dict


class ImageIngestor:
    """
    Manages the ingestion and normalization of professional image formats.
    """

    @staticmethod
    def load_image(file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Detects file type and routes to the appropriate loader with metadata normalization.
        
        Args:
            file_path (str): Absolute path to the image file.
            
        Returns:
            Tuple[np.ndarray, Dict]: The image as a float32 NumPy array and a 
                                     dictionary of normalized metadata.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        raw_extensions = ['.cr2', '.arw', '.nef', '.dng', '.orf']
        
        try:
            if ext in raw_extensions:
                pixels, meta = ImageIngestor._load_raw(file_path)
            else:
                pixels, meta = ImageIngestor._load_oiio(file_path)
            
            # Final safety check for float32 bit-depth
            if pixels.dtype != np.float32:
                pixels = pixels.astype(np.float32)
                
            return pixels, meta
            
        except Exception as e:
            raise IOError(f"Failed to ingest {file_path}: {str(e)}")

    @staticmethod
    def _load_oiio(file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Uses OpenImageIO to load EXR, DPX, TIFF, etc.
        """
        input_file = oiio.ImageInput.open(file_path)
        if not input_file:
            raise IOError(f"OpenImageIO could not open file: {oiio.geterror()}")

        spec = input_file.spec()
        
        # Force read as float32 to maintain signal integrity
        pixels = input_file.read_image(format=oiio.FLOAT)
        input_file.close()

        # Consolidate metadata for the ColorEngine mapper
        metadata = {
            "width": spec.width,
            "height": spec.height,
            "channels": spec.nchannels,
            "file_format": spec.format_name(),
            "colorspace_hint": spec.get_string_attribute("oiio:ColorSpace", "Unknown"),
            "raw_metadata": {attr.name: attr.value for attr in spec.extra_attribs}
        }

        # Handle multi-channel files by extracting only RGB
        if pixels.shape[2] > 3:
            pixels = pixels[:, :, :3]

        return pixels, metadata

    @staticmethod
    def _load_raw(file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Uses Rawpy to process Camera RAW files into a linear buffer.
        """
        with rawpy.imread(file_path) as raw:
            # Linear post-processing (Gamma 1.0) to preserve sensor response
            rgb_linear = raw.postprocess(
                use_camera_wb=False, 
                no_auto_bright=True, 
                output_bps=16, 
                gamma=(1,1)
            )
            
            # Normalize 16-bit int [0, 65535] to float32 [0.0, 1.0]
            pixels = rgb_linear.astype(np.float32) / 65535.0
            
            metadata = {
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "channels": 3,
                "file_format": "Camera RAW",
                "colorspace_hint": "Raw",
                "raw_metadata": {
                    "iso": getattr(raw, 'iso', 'Unknown'),
                    "shutter": getattr(raw, 'shutter', 'Unknown'),
                    "aperture": getattr(raw, 'aperture', 'Unknown'),
                    "camera_make": getattr(raw, 'camera_make', 'Unknown').decode('utf-8', 'ignore'),
                    "camera_model": getattr(raw, 'camera_model', 'Unknown').decode('utf-8', 'ignore')
                }
            }
            
            return pixels, metadata

    @staticmethod
    def validate_signal_range(pixels: np.ndarray):
        """Checks for illegal signal values or clipping."""
        p_max = np.max(pixels)
        if p_max > 1.0:
            print(f"[Warning] HDR values detected (Max: {p_max}). Audit space must be Scene-Linear.")
        elif p_max < 0.01:
            print("[Warning] Extremely low signal detected. Possible underexposure or load error.")