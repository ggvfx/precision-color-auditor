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
        # Safety check: if target is black and observed isn't, dE is invalid for audit
        if np.all(target_rgb == 0): return 0.0
        return float(delta_E(observed_rgb, target_rgb, method='CIE 2000'))

    def _identify_neutrals_by_signal(self, patches: list[ColorPatch]) -> list[int]:
        """Fallback: Identifies patches with the lowest saturation as 'Neutral'."""
        saturations = []
        for p in patches:
            rgb = p.observed_rgb
            sat = np.max(rgb) - np.min(rgb) # Simplified saturation check
            saturations.append(sat)
        
        # Return indices of the 6 least saturated patches
        return np.argsort(saturations)[:6].tolist()

    def calculate_cdl_correction(self, audit_result: AuditResult) -> AuditResult:
        patch_count = len(audit_result.patches)
        signature = settings.get_signature(patch_count)
        
        if signature:
            neutral_indices = signature["neutral_indices"]
        else:
            # Fallback: Guess neutrals by finding the lowest saturation patches
            neutral_indices = self._identify_neutrals_by_signal(audit_result.patches)
        
        obs_neutrals = []
        targ_neutrals = []

        for idx in neutral_indices:
            patch = audit_result.patches[idx]
            obs_neutrals.append(patch.observed_rgb)
            targ_neutrals.append(patch.target_rgb)

        obs = np.array(obs_neutrals)
        targ = np.array(targ_neutrals)

        # Derives Slope (Gain) and Offset (Lift) for neutralization
        slope = np.mean(targ, axis=0) / (np.mean(obs, axis=0) + 1e-6)
        offset = targ[-1] - (obs[-1] * slope)

        audit_result.slope = np.clip(slope, 0.0, 4.0)
        audit_result.offset = np.clip(offset, -1.0, 1.0)
        audit_result.power = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        return audit_result

    def perform_audit(self, file_path: str, sampled_patches: list[ColorPatch]) -> AuditResult:
        """
        Audits the digital signal against mathematical targets to identify drift.
        """
        results = AuditResult(file_path=file_path)

        # 1. Identify the chart and map target data to the discovered patches
        signature = settings.get_signature(len(sampled_patches))
        
        # TODO: In Phase 2, we will fetch actual target RGBs from reference JSONs.
        
        total_de = 0.0
        max_de = 0.0

        for patch in sampled_patches:
            de = self.calculate_delta_e(patch.observed_rgb, patch.target_rgb)
            total_de += de
            max_de = max(max_de, de)
            results.patches.append(patch)

        results.delta_e_mean = total_de / len(sampled_patches) if sampled_patches else 0.0
        results.delta_e_max = max_de
        results.is_pass = results.delta_e_mean <= settings.tolerance_threshold
        
        # Triggers failure flags based on User-defined Tolerance Box
        results.is_pass = results.delta_e_mean <= settings.tolerance_threshold

        return self.calculate_cdl_correction(results)