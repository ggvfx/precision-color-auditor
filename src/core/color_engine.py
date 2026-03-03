"""
Precision Color Auditor - OpenColorIO Management & Transformation
Manages the OCIO v2.x lifecycle and enforces linear-only audit spaces.
"""

import PyOpenColorIO as OCIO
import numpy as np
from core.config import settings

class ColorEngine:
    def __init__(self):
        self.config = None
        self.initialize_config()

    def initialize_config(self):
        """Loads the OCIO configuration from the path defined in settings."""
        config_path = str(settings.current_ocio_path)
        try:
            self.config = OCIO.Config.CreateFromFile(config_path)
            OCIO.SetCurrentConfig(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to load OCIO config at {config_path}: {e}")

    def get_input_spaces(self) -> list[str]:
        """Returns all available color spaces for the source selection."""
        if not self.config:
            return []
        return [space.getName() for space in self.config.getColorSpaces()]

    def get_linear_audit_spaces(self) -> list[str]:
        """
        Populates the Audit Space dropdown with only Linear/Scene-Linear spaces.
        Filters based on OCIO 'family' or bit-depth characteristics.
        """
        if not self.config:
            return []
        
        linear_spaces = []
        for space in self.config.getColorSpaces():
            name = space.getName().lower()
            family = space.getFamily().lower()
            
            # Filter logic: Look for 'linear' in the name or family tag
            if "linear" in name or "linear" in family or "acescg" in name:
                linear_spaces.append(space.getName())
        
        # Ensure ACEScg is at the top of the list if it exists
        if "ACES - ACEScg" in linear_spaces:
            linear_spaces.insert(0, linear_spaces.pop(linear_spaces.index("ACES - ACEScg")))
            
        return linear_spaces

    def map_metadata_to_space(self, metadata: dict) -> str:
        """
        Refined to prioritize Ingest hints and handle RAW vs Generic fallbacks.
        """
        # Normalize Camera Info for the Report
        raw_info = metadata.get("raw_metadata", {})
        metadata["camera_make"] = raw_info.get("camera_make", "Unknown")
        metadata["camera_model"] = raw_info.get("camera_model", "Unknown")

        available = self.get_input_spaces()
        
        # 1. Handle RAW files specifically 
        # (rawpy output is Linear Rec.709 primaries by default)
        if metadata.get("is_raw"):
            if "Utility - Linear - Rec.709" in available:
                return "Utility - Linear - Rec.709"
            if "Linear" in available:
                return "Linear"

        # 2. Check the explicit hint generated in ingest.py (e.g., 'sRGB', 'Linear')
        hint = metadata.get("colorspace_hint")
        if hint in available:
            return hint
            
        # 3. Fallback to common industry metadata keys
        potential_keys = ["Colorspace", "ocioconfig", "interpretation"]
        for key in potential_keys:
            val = metadata.get(key, "")
            if val in available:
                return val
        
        # 4. Extension-based guessing for stripped files
        fmt = metadata.get("file_format", "").upper()
        if (fmt == "EXR" or "EXR" in fmt) and "Linear" in available:
            return "Linear"
        if fmt in ["JPG", "JPEG", "PNG"] and "sRGB" in available:
            return "sRGB"
        
        # Final fallback
        return "Raw"

    def transform_buffer(self, pixel_buffer: np.ndarray, input_space: str, audit_space: str) -> np.ndarray:
        """
        Transforms pixel data from Source to Linear Audit space.
        """
        try:
            processor = self.config.getProcessor(input_space, audit_space)
            cpu_processor = processor.getDefaultCPUProcessor()
            return cpu_processor.applyRGB(pixel_buffer)
        except Exception as e:
            print(f"Transform Error: {e}")
            return pixel_buffer