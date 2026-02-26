"""
Precision Color Auditor - Global Configuration
Manages application state, OCIO paths, and user-defined audit thresholds.
Acts as a central source of truth for all modules.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    """
    app_root: Path = Path(__file__).parent.parent.parent
    
    # OCIO Configurations
    default_ocio_path: Path = field(init=False)
    current_ocio_path: Path = field(init=False)
    
    # Audit Thresholds
    tolerance_threshold: float = 2.0  # Industry standard dE2000 'Noticeable' limit
    sample_size: int = 32            # 32x32 pixel average to ignore sensor noise
    
    # Export Settings
    output_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize dynamic paths and ensure directories exist."""
        # Define internal resource paths
        self.default_ocio_path = self.app_root / "src" / "resources" / "ocio" / "config.ocio"
        self.current_ocio_path = self.default_ocio_path
        
        # Define default export path
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


# Global Singleton Instance
# Every module should import this instance: 'from core.config import settings'
settings = Settings()