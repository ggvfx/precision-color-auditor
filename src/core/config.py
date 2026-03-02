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


@dataclass
class Settings:
    """
    Global settings for the Precision Color Auditor.
    
    Attributes:
        app_root (Path): The root directory of the application.
        default_ocio_path (Path): Path to the bundled ACES 1.3 config.
        current_ocio_path (Path): The active OCIO config (defaults to bundled).
        tolerance_threshold (float): Delta E (dE2000) limit before flagging a fail.
        sample_size (int): The square size (in pixels) for mean patch sampling.
        output_dir (Path): Default location for reports and CDL exports.
        crops_dir (Path): Location for rectified QC images (Official Output).
        rectified_size (Tuple): Standard resolution for the warped chart crop.
        active_chart_type (str): Current template key (defaulting to 'macbeth_24').
    """
    app_root: Path = Path(__file__).parent.parent.parent
    
    # OCIO Configurations
    default_ocio_path: Path = field(init=False)
    current_ocio_path: Path = field(init=False)
    
    # Audit Thresholds
    tolerance_threshold: float = 2.0  # Industry standard dE2000 'Noticeable' limit
    sample_size: int = 32            # 32x32 pixel average to ignore sensor noise

    # Rectification & Topology Settings
    # Standardizing the warp target ensures consistent sampling math
    rectified_size: Tuple[int, int] = (1200, 800) 
    active_chart_type: str = "kodak_gray_plus"

    # Template Library
    chart_templates: Dict[str, Dict] = field(default_factory=lambda: {
        "macbeth_24": {
            "label": "Macbeth 24-Patch",
            "topology": "grid", # Standard 6x4 math
            "grid": (6, 4), 
            "neutral_indices": list(range(18, 24)),
            "target_space": "ACEScg",
            "detection_prompt": "macbeth color calibration chart, 6x4 rectangular color samples surrounded by black borders"
        },
        "kodak_gray_plus": {
            "label": "Kodak Gray Card Plus",
            "topology": "anchored", # Custom coordinate map
            "target_space": "ACEScg",
            "detection_prompt": "Kodak Gray Card Plus, large central gray square with white and black rectangular patches on the left and right sides",
            # Normalized coordinates (0.0 to 1.0) for the 1200x800 rectified crop
            "anchors": {
                "main_gray": {"pos": (0.5, 0.5), "label": "18% Gray"},
                "left_top_black": {"pos": (0.2, 0.25), "label": "Left Black"},
                "left_bottom_white": {"pos": (0.2, 0.75), "label": "Left White"},
                "right_top_white": {"pos": (0.8, 0.25), "label": "Right White"},
                "right_bottom_black": {"pos": (0.8, 0.75), "label": "Right Black"}
            },
            "neutral_indices": ["main_gray", "left_bottom_white", "right_top_white"] 
        }
    })

    # Manual Draw Override Settings
    use_manual_locator: bool = False
    manual_corners: Optional[np.ndarray] = None

    # New Official Output Directory
    crops_dir: Path = field(init=False)
    
    # Export Settings
    output_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize dynamic paths and ensure directories exist."""
        self.default_ocio_path = self.app_root / "src" / "resources" / "ocio" / "config.ocio"
        self.current_ocio_path = self.default_ocio_path
        
        # Paths for Official Outputs
        self.output_dir = self.app_root / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update_ocio_config(self, custom_path: str):
        """
        Updates the active OCIO config if the file exists.
        
        Args:
            custom_path (str): Path to the user-provided .ocio file.
        """
        path = Path(custom_path)
        if path.exists() and path.suffix == ".ocio":
            self.current_ocio_path = path
        else:
            raise FileNotFoundError(f"Invalid OCIO config path: {custom_path}")
        
    def get_current_template(self) -> Dict:
        """Returns the geometric template for the active chart type."""
        return self.chart_templates.get(self.active_chart_type)
    
    def get_signature(self, patch_count: int):
        """Returns the chart metadata based on the number of patches found."""
        signatures = {
            24: {"label": "Macbeth 24", "rows": 4, "cols": 6},
            12: {"label": "Grayscale 12", "rows": 1, "cols": 12},
            # Add more as we expand
        }
        return signatures.get(patch_count)


# Global Singleton Instance
# Every module should import this instance: 'from core.config import settings'
settings = Settings()