"""
Precision Color Auditor - Core Data Models
Defines the structured data objects for color patch metadata, sampling results, 
and technical audit metrics used across the DI/VFX pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime
from enum import Enum, auto


@dataclass(frozen=False)
class ColorPatch:
    """
    Represents a single sampled patch from a physical color reference chart.
    
    Attributes:
        name (str): The common name of the patch (e.g., 'Neutral 5', 'Dark Skin').
        observed_rgb (np.ndarray): The mean RGB values sampled from the image (Scene-Linear).
        target_rgb (np.ndarray): The mathematical 'Ideal' RGB values for the specific color space.
        local_center (Tuple[int, int]): The (x, y) center coordinate relative to the rectified crop.
        index (int): The patch index (0-23 for Macbeth, 0-5 for Greyscale).
    """
    name: str
    observed_rgb: np.ndarray
    target_rgb: np.ndarray
    local_center: Tuple[int, int]
    index: int
    delta_e: float = 0.0
    # Geometric Health
    # Stores the standard deviation of pixels in the sample box
    sample_variance: float = 0.0
    is_contaminated: bool = False

    def __post_init__(self):
        """Validates that RGB data is stored as high-precision floats."""
        if self.observed_rgb.dtype != np.float32 and self.observed_rgb.dtype != np.float64:
            object.__setattr__(self, 'observed_rgb', self.observed_rgb.astype(np.float32))
        if self.target_rgb.dtype != np.float32 and self.target_rgb.dtype != np.float64:
            object.__setattr__(self, 'target_rgb', self.target_rgb.astype(np.float32))

class AuditStatus(Enum):
    IDLE = auto()          # Just loaded, no math done yet
    PENDING = auto()       # Currently in the AI/Sampler queue
    COMPLETE = auto()      # Successfully processed
    MANUAL_EDIT = auto()   # User moved corners; needs a re-sample
    FAILED = auto()        # AI couldn't find a chart

@dataclass
class AuditResult:
    file_path: str
    template_name: str = "macbeth_24"
    status: AuditStatus = AuditStatus.IDLE
    # --- Per-Task Overrides ---
    # These default to the Global settings if None
    input_space: Optional[str] = None  # e.g. "ARRI LogC4"
    audit_space: Optional[str] = None # e.g. "ACEScg"
    display_space: Optional[str] = None # e.g. "rec709"
    # --------------------------
    ai_reasoning: str = ""
    corners: Optional[np.ndarray] = None 
    rectified_path: Optional[str] = None
    rectified_buffer: Optional[np.ndarray] = None
    delta_e_mean: float = 0.0
    delta_e_max: float = 0.0
    is_pass: bool = False

    camera_make: str = "Unknown"
    camera_model: str = "Unknown"

    # New Fields for Intent and Integrity
    analysis_intent: str = "neutralize" # Snapshot of setting used
    alignment_integrity: float = 1.0     # 0.0 to 1.0 (1.0 = Perfect)
    integrity_warning: bool = False      # UI Flag for "Check Corners"
    
    # ASC-CDL Neutralization Values
    slope: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    power: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    sat: float = 1.0

    # Color Matrix (Future-proofing for Tier 4)
    # Identity matrix by default (no change)
    matrix_3x3: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    
    patches: List[ColorPatch] = field(default_factory=list)
    timestamp: Optional[str] = None

    def get_sop_summary(self) -> str:
        return (f"Slope: {self.slope} | Offset: {self.offset} | "
                f"Power: {self.power} | Sat: {self.sat}")
    
    def get_neutral_patches(self) -> List[ColorPatch]:
        """Returns patches flagged as neutral in the active template."""
        from core.config import settings
        template = settings.get_current_template()
        neutral_indices = template.neutral_indices
        
        return [
            p for p in self.patches 
            if p.index in neutral_indices or p.name.replace("Patch_", "") in neutral_indices
        ]
    
@dataclass
class AuditTask:
    """
    The Session Manager's container for tracking an image through the pipeline.
    This links a physical file to its mathematical AuditResult and its UI state.
    """
    task_id: str
    file_path: str
    status: str = "Pending"  # Options: Pending, Processing, Success, Failed, Manual_Required
    
    # User Overrides (Set by the UI)
    manual_corners: Optional[np.ndarray] = None
    input_space_override: Optional[str] = None
    
    # The Payload
    result: Optional[AuditResult] = None
    error_message: Optional[str] = None