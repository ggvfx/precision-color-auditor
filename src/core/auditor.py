"""
Precision Color Auditor - Signal Analysis & Neutralization Logic
Calculates Delta E error metrics and determines required ASC-CDL offsets[cite: 15].
Validates sampled patch data against mathematical 'Ground Truth' ideals[cite: 3, 15].
"""

import numpy as np
from colour import colour_checkers, delta_E
from core.models import ColorPatch, AuditResult
from core.config import settings

class Auditor:
    """
    Core math logic comparing sampled pixels to target color space ideals[cite: 15].
    """

    def __init__(self):
        # Uses the 'colour-science' library for industry-standard math
        self.target_data = colour_checkers.CCS_COLOURCHECKERS['ColorChecker24'].data

    def calculate_delta_e(self, observed_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
        """Calculates Delta E 2000 (dE2000) for high-precision error reporting."""
        return float(delta_E(observed_rgb, target_rgb, method='CIE 2000'))

    def calculate_cdl_correction(self, audit_result: AuditResult) -> AuditResult:
        """
        Generates the mathematical neutralization data in ASC-CDL format.
        Focuses on the 6 neutral patches to identify color drift and exposure errors.
        """
        # Neutral patches are indices 18-23 on a standard Macbeth chart
        neutral_indices = range(18, 24)
        obs_neutrals = []
        targ_neutrals = []

        for idx in neutral_indices:
            # Matches observed signal against the 'Ideal' targets
            patch = audit_result.patches[idx]
            obs_neutrals.append(patch.observed_rgb)
            targ_neutrals.append(patch.target_rgb)

        obs = np.array(obs_neutrals)
        targ = np.array(targ_neutrals)

        # Derives Slope (Gain) and Offset (Lift) for neutralization
        slope = np.mean(targ, axis=0) / np.mean(obs, axis=0)
        offset = targ[-1] - (obs[-1] * slope)

        audit_result.slope = np.clip(slope, 0.0, 4.0)
        audit_result.offset = np.clip(offset, -1.0, 1.0)
        audit_result.power = np.array([1.0, 1.0, 1.0])
        
        return audit_result

    def perform_audit(self, file_path: str, sampled_patches: list[ColorPatch]) -> AuditResult:
        """
        Audits the digital signal against mathematical targets to identify drift.
        """
        results = AuditResult(file_path=file_path)
        total_de = 0.0
        max_de = 0.0

        for patch in sampled_patches:
            de = self.calculate_delta_e(patch.observed_rgb, patch.target_rgb)
            total_de += de
            max_de = max(max_de, de)
            results.patches.append(patch)

        results.delta_e_mean = total_de / len(sampled_patches)
        results.delta_e_max = max_de
        
        # Triggers failure flags based on User-defined Tolerance Box
        results.is_pass = results.delta_e_mean <= settings.tolerance_threshold

        return self.calculate_cdl_correction(results)