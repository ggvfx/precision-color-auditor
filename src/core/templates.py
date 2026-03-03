from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

@dataclass(frozen=True)
class ChartTemplate:
    """
    Blueprint for a physical color reference chart.
    
    orientation_anchor: A Tuple (brighter_index, darker_index) used to 
        detect if a chart is 180° upside down. For Macbeth, this is 
        usually (White Patch, Black Patch). Set to None if symmetrical.
    """
    name: str
    label: str
    topology: str  # 'grid' or 'anchored'
    rectified_size: Tuple[int, int]
    inset_margin: float
    sample_size: int
    target_space: str
    detection_prompt: str
    grid: Optional[Tuple[int, int]] = None
    anchors: Optional[Dict[str, Dict[str, Any]]] = None
    neutral_indices: List[Any] = field(default_factory=list)
    orientation_anchor: Optional[Tuple[Any, Any]] = None

# Chart Templates
MACBETH_24 = ChartTemplate(
    name="macbeth_24",
    label="Macbeth 24-Patch",
    topology="grid",
    grid=(6, 4),
    rectified_size=(1200, 800),
    inset_margin=0.03,
    sample_size=32,
    target_space="ACEScg",
    detection_prompt="macbeth color calibration chart, 6x4 rectangular color samples surrounded by black borders",
    neutral_indices=list(range(18, 24)),
    orientation_anchor=(18, 5) # White patch (18) must be brighter than Teal patch (5)
)

KODAK_GRAY_PLUS = ChartTemplate(
    name="kodak_gray_plus",
    label="Kodak Gray Card Plus",
    topology="anchored",
    rectified_size=(1200, 900),
    inset_margin=0.02,
    sample_size=64,
    target_space="ACEScg",
    detection_prompt="Kodak Gray Card Plus, large central gray square...",
    anchors={
        "main_gray": {"pos": (0.5, 0.5), "label": "18% Gray"},
        # ... other anchors
    },
    neutral_indices=["main_gray", "left_bottom_white", "right_top_white"],
    orientation_anchor=None
)

CHART_LIBRARY = {
    "macbeth_24": MACBETH_24,
    "kodak_gray_plus": KODAK_GRAY_PLUS
}