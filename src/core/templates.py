from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

@dataclass(frozen=True)
class ChartTemplate:
    """
    Blueprint for a physical color reference chart.
    """
    name: str
    label: str
    topology: str  # 'grid' or 'anchored'
    analysis_mode: str  # 'gain', 'anchors', 'ramp', or 'color'
    rectified_size: Tuple[int, int]
    inset_margin: float
    sample_size: int
    reference_space: str
    detection_prompt: str
    grid: Optional[Tuple[int, int]] = None
    anchors: Optional[Dict[str, Dict[str, Any]]] = None
    neutral_indices: List[Any] = field(default_factory=list)
    orientation_anchor: Optional[Tuple[Any, Any]] = None
    color_targets: Dict[Any, List[float]] = field(default_factory=dict)

# 1. MACBETH 24 (Tier 4: Color Matrix)
MACBETH_24 = ChartTemplate(
    name="macbeth_24",
    label="Macbeth 24-Patch",
    topology="grid",
    analysis_mode="color",  # Full Color Profiling
    grid=(6, 4),
    rectified_size=(1200, 800),
    inset_margin=0.03,
    sample_size=32,
    reference_space="ACEScg",
    detection_prompt="macbeth color calibration chart, 6x4 rectangular color samples surrounded by black borders",
    neutral_indices=list(range(18, 24)),
    orientation_anchor=(18, 5),
    color_targets={
        0: [0.118, 0.082, 0.065], 1: [0.398, 0.285, 0.219], 2: [0.153, 0.186, 0.260],
        3: [0.100, 0.124, 0.063], 4: [0.245, 0.231, 0.355], 5: [0.133, 0.315, 0.283],
        6: [0.551, 0.243, 0.057], 7: [0.083, 0.116, 0.334], 8: [0.354, 0.108, 0.129],
        9: [0.076, 0.052, 0.110], 10: [0.331, 0.470, 0.108], 11: [0.528, 0.351, 0.068],
        12: [0.038, 0.051, 0.222], 13: [0.117, 0.240, 0.085], 14: [0.183, 0.047, 0.052],
        15: [0.672, 0.547, 0.048], 16: [0.338, 0.134, 0.334], 17: [0.049, 0.169, 0.310],
        # Neutral Ramp (Row 4)
        18: [0.865, 0.865, 0.865], 19: [0.589, 0.589, 0.589], 20: [0.360, 0.360, 0.360],
        21: [0.198, 0.198, 0.198], 22: [0.090, 0.090, 0.090], 23: [0.031, 0.031, 0.031]
    }
)

# 2. KODAK GRAY CARD PLUS (Tier 2: Anchor Set)
KODAK_GRAY_PLUS = ChartTemplate(
    name="kodak_gray_plus",
    label="Kodak Gray Card Plus",
    topology="anchored",
    analysis_mode="anchors",  # Specific named patch calibration
    rectified_size=(1200, 900),
    inset_margin=0.02,
    sample_size=64,
    reference_space="ACEScg",
    detection_prompt="Kodak Gray Card Plus, large central gray square...",
    anchors={
        "main_gray":       {"pos": (0.50, 0.50), "label": "18% Gray"},
        "top_left_black":  {"pos": (0.08, 0.25), "label": "Black Patch TL"},
        "bot_left_white":  {"pos": (0.08, 0.75), "label": "White Patch BL"},
        "top_right_white": {"pos": (0.92, 0.25), "label": "White Patch TR"},
        "bot_right_black": {"pos": (0.92, 0.75), "label": "Black Patch BR"},
    },
    # Synchronized names with the anchors above
    neutral_indices=["main_gray", "bot_left_white", "top_right_white", "top_left_black", "bot_right_black"],
    orientation_anchor=None,
    color_targets={
        "main_gray":       [0.180, 0.180, 0.180],
        "top_left_black":  [0.031, 0.031, 0.031],
        "bot_left_white":  [0.865, 0.865, 0.865],
        "top_right_white": [0.865, 0.865, 0.865],
        "bot_right_black": [0.031, 0.031, 0.031]
    }
)

# 3. 11-STEP GRAYSCALE (Tier 3: Neutral Ramp)
# (Placeholder for future addition)
GRAYSCALE_11 = ChartTemplate(
    name="grayscale_11",
    label="11-Step Neutral Ramp",
    topology="grid",
    analysis_mode="ramp",  # Sequence profiling
    grid=(11, 1),
    rectified_size=(1100, 200),
    inset_margin=0.05,
    sample_size=32,
    reference_space="ACEScg",
    detection_prompt="linear grayscale ramp with 11 steps from black to white",
    neutral_indices=list(range(11)),
    color_targets={i: [v, v, v] for i, v in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
)

# 4. SINGLE 18% GRAY (Tier 1: Single Point)
GRAY_CARD = ChartTemplate(
    name="gray_card",
    label="Standard Gray Card",
    topology="anchored",
    analysis_mode="gain",  # Mid-gray balance only
    rectified_size=(800, 800),
    inset_margin=0.1,
    sample_size=128,
    reference_space="ACEScg",
    detection_prompt="solid neutral gray card",
    anchors={"center": {"pos": (0.5, 0.5), "label": "18% Gray"}},
    neutral_indices=["center"],
    color_targets={"center": [0.18, 0.18, 0.18]}
)

CHART_LIBRARY = {
    "macbeth_24": MACBETH_24,
    "kodak_gray_plus": KODAK_GRAY_PLUS,
    "grayscale_11": GRAYSCALE_11,
    "gray_card": GRAY_CARD
}