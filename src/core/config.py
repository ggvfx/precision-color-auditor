"""
Precision Color Auditor - Global Configuration
Manages application state, OCIO paths, and user-defined audit thresholds.
Acts as a central source of truth for all modules.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

from .templates import CHART_LIBRARY, ChartTemplate

@dataclass
class Settings:
    """
    Global settings and session-level defaults.
    """
    app_root: Path = Path(__file__).parent.parent.parent
    
    # OCIO Configurations
    default_ocio_path: Path = field(init=False)
    current_ocio_path: Path = field(init=False)
    
    # Default Input/Output Colorspaces (The 'Global' default)
    default_input_space: str = "ACES - ACEScg"
    default_display_space: str = "sRGB - Texture"
    
    # Audit Thresholds
    tolerance_threshold: float = 2.0
    sample_size: int = 32

    # Rectification & Topology Settings
    rectified_size: Tuple[int, int] = (1200, 800) 
    active_chart_type: str = "macbeth_24"

    # Directory Management
    output_dir: Path = field(init=False)
    session_logs_dir: Path = field(init=False)

    # Analysis Mode
    # Options: "neutralize" (Fix the image) or "match_grade" (Extract the look)
    analysis_intent: str = "neutralize" 
    
    # Alignment Integrity (Geometric Health)
    # Threshold for standard deviation in a patch sample. 
    # Values above this suggest the sample box hit a bezel or border.
    integrity_threshold: float = 0.05

    def __post_init__(self):
        """Initialize dynamic paths and ensure directories exist."""
        self.default_ocio_path = self.app_root / "src" / "resources" / "ocio" / "config.ocio"
        self.current_ocio_path = self.default_ocio_path
        
        # Paths for Official Outputs & Logs
        self.output_dir = self.app_root / "exports"
        self.session_logs_dir = self.app_root / "logs"
        
        # Ensure all required directories exist
        for d in [self.output_dir, self.session_logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_current_template(self) -> ChartTemplate:
        """Returns the active ChartTemplate object."""
        return CHART_LIBRARY.get(self.active_chart_type, CHART_LIBRARY["macbeth_24"])

    def update_ocio_config(self, custom_path: str):
        """Updates the active OCIO config if the file exists."""
        path = Path(custom_path)
        if path.exists() and path.suffix == ".ocio":
            self.current_ocio_path = path
        else:
            raise FileNotFoundError(f"Invalid OCIO config path: {custom_path}")
    
    def get_signature(self, patch_count: int):
        """Returns the chart metadata based on the number of patches found."""
        signatures = {
            24: {"label": "Macbeth 24", "rows": 4, "cols": 6},
            12: {"label": "Grayscale 12", "rows": 1, "cols": 12},
        }
        return signatures.get(patch_count)

# Global Singleton Instance
settings = Settings()