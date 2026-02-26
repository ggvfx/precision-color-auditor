"""
Precision Color Auditor - Core Data Models
Defines the structured data objects for color patch metadata, sampling results, 
and technical audit metrics used across the DI/VFX pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class ColorPatch:
    """
    Represents a single sampled patch from a physical color reference chart.
    
    Attributes:
        name (str): The common name of the patch (e.g., 'Neutral 5', 'Dark Skin').
        observed_rgb (np.ndarray): The mean RGB values sampled from the image (Scene-Linear).
        target_rgb (np.ndarray): The mathematical 'Ideal' RGB values for the specific color space.
        coordinates (Tuple[int, int, int, int]): The ROI bounding box (x, y, w, h).
        index (int): The patch index (0-23 for Macbeth, 0-5 for Greyscale).
    """
    name: str
    observed_rgb: np.ndarray
    target_rgb: np.ndarray
    coordinates: Tuple[int, int, int, int]
    index: int

    def __post_init__(self):
        """Validates that RGB data is stored as high-precision floats."""
        if self.observed_rgb.dtype != np.float32 and self.observed_rgb.dtype != np.float64:
            object.__setattr__(self, 'observed_rgb', self.observed_rgb.astype(np.float32))
        if self.target_rgb.dtype != np.float32 and self.target_rgb.dtype != np.float64:
            object.__setattr__(self, 'target_rgb', self.target_rgb.astype(np.float32))


@dataclass
class AuditResult:
    """
    Holds the calculated error metrics and neutralization data for an audited image.
    
    Attributes:
        file_path (str): Source path of the audited image.
        delta_e_mean (float): The average Delta E (dE2000) across all sampled patches.
        delta_e_max (float): The highest recorded Delta E error in the set.
        is_pass (bool): Whether the audit falls within the user-defined tolerance.
        
        # ASC-CDL Neutralization Values
        slope (np.ndarray): RGB Slope values (default [1.0, 1.0, 1.0]).
        offset (np.ndarray): RGB Offset values (default [0.0, 0.0, 0.0]).
        power (np.ndarray): RGB Power values (default [1.0, 1.0, 1.0]).
        sat (float): Global Saturation value (default 1.0).
        
        patches (List[ColorPatch]): List of individual patch data for granular reporting.
        timestamp (str): ISO formatted string of the audit execution time.
    """
    file_path: str
    delta_e_mean: float = 0.0
    delta_e_max: float = 0.0
    is_pass: bool = False
    
    slope: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    power: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    sat: float = 1.0
    
    patches: List[ColorPatch] = field(default_factory=list)
    timestamp: Optional[str] = None

    def get_sop_summary(self) -> str:
        """Returns a formatted string of the CDL SOP values for logging."""
        return (f"Slope: {self.slope} | Offset: {self.offset} | "
                f"Power: {self.power} | Sat: {self.sat}")