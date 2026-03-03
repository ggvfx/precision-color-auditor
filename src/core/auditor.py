"""
Precision Color Auditor - Signal Analysis & Neutralization Logic
Standardized to template-based targets and ACEScg workflow.
"""

import numpy as np
import colour # Professional standard for Delta E math
from core.models import ColorPatch, AuditResult
from core.config import settings

class Auditor:
    """
    Core math logic comparing sampled pixels to template-defined ideals.
    """

    def __init__(self):
        # We rely on the template injected during sampling
        pass

    def calculate_delta_e(self, observed_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
        """
        Calculates Delta E 2000. 
        Note: Requires conversion from ACEScg (AP1) to Lab space.
        """
        if np.all(target_rgb == 0): return 0.0
        
        # Convert ACEScg -> XYZ -> Lab for accurate Delta E calculation
        # 'ACEScg' is the native AP1/Linear space from our ColorEngine
        obs_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(observed_rgb, 'ACEScg'))
        targ_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(target_rgb, 'ACEScg'))
        
        return float(colour.delta_E(obs_lab, targ_lab, method='CIE 2000'))

    def calculate_cdl_correction(self, audit_result: AuditResult) -> AuditResult:
        """
        Derives Slope (Gain) and Offset (Lift) for neutralization 
        based on the active template's neutral_indices.
        """
        template = settings.get_current_template()
        neutral_indices = template.neutral_indices
        
        obs_neutrals = []
        targ_neutrals = []

        # Find patches that are flagged as Neutral in the template
        for patch in audit_result.patches:
            # Check if patch index or its key (for anchored charts) is in neutrals
            clean_name = patch.name.replace("Patch_", "")
            if patch.index in neutral_indices or clean_name in neutral_indices:
                obs_neutrals.append(patch.observed_rgb)
                targ_neutrals.append(patch.target_rgb)

        if not obs_neutrals:
            return audit_result

        obs = np.array(obs_neutrals)
        targ = np.array(targ_neutrals)

        # Linear correction: targ = (obs * slope) + offset
        # Derived from mean of neutral ramp to minimize noise impact
        slope = np.mean(targ, axis=0) / (np.mean(obs, axis=0) + 1e-6)
        
        # Anchor the offset to the darkest neutral (the last one in the ramp)
        offset = targ[-1] - (obs[-1] * slope)

        audit_result.slope = np.clip(slope, 0.0, 4.0)
        audit_result.offset = np.clip(offset, -1.0, 1.0)
        audit_result.power = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        return audit_result

    def perform_audit(self, file_path: str, sampled_patches: list[ColorPatch]) -> AuditResult:
        """
        Audits the digital signal against template targets to identify drift.
        """
        results = AuditResult(file_path=file_path)
        
        total_de = 0.0
        max_de = 0.0

        for patch in sampled_patches:
            # calculate_delta_e handles the ACEScg-to-Lab math
            de = self.calculate_delta_e(patch.observed_rgb, patch.target_rgb)
            
            # Store error on the patch for per-patch reporting
            patch.delta_e = de 
            
            total_de += de
            max_de = max(max_de, de)
            results.patches.append(patch)

        # Calculate metrics
        count = len(sampled_patches)
        results.delta_e_mean = total_de / count if count > 0 else 0.0
        results.delta_e_max = max_de
        
        # Pass/Fail based on the user's global tolerance setting
        results.is_pass = results.delta_e_mean <= settings.tolerance_threshold

        # Finally, calculate the CDL values needed to "fix" the image
        return self.calculate_cdl_correction(results)