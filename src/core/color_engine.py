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
        Filters based on OCIO 'family' or bit-depth characteristics[cite: 7, 21].
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
        Attempts to find a matching OCIO space based on image metadata[cite: 4].
        """
        # Look for common metadata keys from OIIO/Rawpy
        potential_keys = ["Colorspace", "ocioconfig", "interpretation"]
        available = self.get_input_spaces()

        for key in potential_keys:
            val = metadata.get(key, "")
            if val in available:
                return val
        
        # Default fallback if no metadata match is found
        return "Raw"

    def transform_buffer(self, pixel_buffer: np.ndarray, input_space: str, audit_space: str) -> np.ndarray:
        """
        Transforms pixel data from Source to Linear Audit space[cite: 15, 21].
        """
        try:
            processor = self.config.getProcessor(input_space, audit_space)
            cpu_processor = processor.getDefaultCPUProcessor()
            return cpu_processor.applyRGB(pixel_buffer)
        except Exception as e:
            print(f"Transform Error: {e}")
            return pixel_buffer