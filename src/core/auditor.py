"""
Precision Color Auditor - Signal Analysis & Neutralization Logic
Standardized to template-based targets and ACEScg workflow.
"""

import numpy as np
import colour
from core.models import ColorPatch, AuditResult
from core.config import settings
from core.templates import CHART_LIBRARY
from datetime import datetime

class Auditor:
    def __init__(self):
        pass

    def calculate_delta_e(self, observed_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
        """Calculates Delta E 2000 in the Audit Space (ACEScg)."""
        if np.all(target_rgb <= 0): return 0.0
        obs_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(observed_rgb, 'ACEScg'))
        targ_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(target_rgb, 'ACEScg'))
        return float(colour.delta_E(obs_lab, targ_lab, method='CIE 2000'))

    def calculate_cdl_correction(self, audit_result: AuditResult) -> AuditResult:
        """
        Dispatches to the correct math based on the template's analysis_mode.
        """
        # Look up the specific template used when this image was sampled
        template = CHART_LIBRARY.get(audit_result.template_name)
        
        # Fallback safety
        if not template:
            print(f"WARNING: Template {audit_result.template_name} not found. Defaulting to Macbeth.")
            template = CHART_LIBRARY["macbeth_24"]

        mode = getattr(template, 'analysis_mode', 'gain')

        # Map the patches for easy lookup
        patch_map = {p.name.replace("Patch_", "").strip().lower(): p for p in audit_result.patches}
        index_map = {p.index: p for p in audit_result.patches}

        if mode == "gain":
            return self._solve_gain(audit_result)
        
        elif mode == "anchors":
            # Tier 2: Kodak - Uses B, G, W for a robust linear fit
            return self._solve_anchors(audit_result, patch_map)
        
        elif mode == "ramp":
            # Tier 3: Grayscale Ramp - Linear fit through sequence
            return self._solve_ramp(audit_result, index_map, template.neutral_indices)
            
        elif mode == "color":
            # Tier 4: Macbeth - CDL + placeholder for Matrix
            return self._solve_color(audit_result, index_map, template.neutral_indices)

        print("WARNING: No mode matched. Returning identity.") 
        return audit_result

    def _get_regression_data(self, obs: np.ndarray, targ: np.ndarray, intent: str):
        """Helper to swap X/Y based on user intent."""
        if intent == "match_grade":
            return targ, obs  # Mapping Target -> Observed (How was this grade made?)
        return obs, targ      # Mapping Observed -> Target (How do I fix this?)

    def _solve_gain(self, result: AuditResult) -> AuditResult:
        """TIER 1: Single patch balance."""
        if not result.patches: return result
        
        obs = result.patches[0].observed_rgb
        targ = result.patches[0].target_rgb
        
        # Apply intent swap
        x, y = self._get_regression_data(obs, targ, result.analysis_intent)
        
        # Slope = Y / X (with epsilon to avoid division by zero)
        result.slope = np.clip(y / (x + 1e-6), 0.0, 4.0).astype(np.float32)
        result.offset = np.zeros(3, dtype=np.float32)
        result.power = np.ones(3, dtype=np.float32)
        
        return result

    def _solve_anchors(self, result: AuditResult, patch_map: dict) -> AuditResult:
        """TIER 2: Kodak / Anchor Set. Linear fit through pinned patches."""
        template = CHART_LIBRARY.get(result.template_name)
        neutral_names = template.neutral_indices

        obs_list, targ_list = [], []
        for name in neutral_names:
            search_key = str(name).strip().lower()
            p = patch_map.get(search_key)
            if p:
                obs_list.append(p.observed_rgb)
                targ_list.append(p.target_rgb)

        if len(obs_list) < 2: return result

        # Apply intent swap to the arrays
        x_data, y_data = self._get_regression_data(np.array(obs_list), np.array(targ_list), result.analysis_intent)

        slopes, offsets = [], []
        for i in range(3):
            m, c = np.polyfit(x_data[:, i], y_data[:, i], 1)
            slopes.append(m)
            offsets.append(c)

        result.slope = np.clip(slopes, 0.0, 4.0).astype(np.float32)
        result.offset = np.clip(offsets, -1.0, 1.0).astype(np.float32)
        return result

    def _solve_ramp(self, result: AuditResult, index_map: dict, neutral_indices: list) -> AuditResult:
        """TIER 3/4: Macbeth/Ramp Style. Regression fit through multiple points."""
        obs_list, targ_list = [], []
        for idx in neutral_indices:
            p = index_map.get(idx)
            if p:
                obs_list.append(p.observed_rgb)
                targ_list.append(p.target_rgb)

        if len(obs_list) < 2: return result

        # Apply intent swap
        x_data, y_data = self._get_regression_data(np.array(obs_list), np.array(targ_list), result.analysis_intent)

        slopes, offsets = [], []
        for i in range(3):
            m, c = np.polyfit(x_data[:, i], y_data[:, i], 1)
            slopes.append(m)
            offsets.append(c)

        result.slope = np.clip(slopes, 0.0, 4.0).astype(np.float32)
        result.offset = np.clip(offsets, -1.0, 1.0).astype(np.float32)
        return result
    
    def _solve_color(self, result: AuditResult, index_map: dict, neutral_indices: list) -> AuditResult:
        """TIER 4: Macbeth. Neutralization CDL + Matrix Profiling."""
        # 1. First, solve the 1D neutralization (Slope/Offset)
        result = self._solve_ramp(result, index_map, neutral_indices)
        
        # 2. Then, solve the 3x3 Color Matrix
        result = self._solve_3x3_matrix(result)
        
        return result
    
    def _solve_3x3_matrix(self, result: AuditResult) -> AuditResult:
        """
        TIER 4: Macbeth Full Profiling.
        Solves for a 3x3 matrix using the 18 color patches.
        """
        # 1. Gather color patches (Indices 0-17 on a Macbeth 24)
        obs_list = []
        targ_list = []
        
        # We only use color patches for the matrix to avoid skewing the neutrals
        for p in result.patches:
            if p.index < 18:
                obs_list.append(p.observed_rgb)
                targ_list.append(p.target_rgb)
        
        if len(obs_list) < 3:
            return result # Need at least 3 colors to solve a 3x3

        # 2. Prepare Data
        # We use the same intent logic as the CDL
        obs = np.array(obs_list)
        targ = np.array(targ_list)
        
        x_data, y_data = self._get_regression_data(obs, targ, result.analysis_intent)

        # 3. Solve for Matrix M using Linear Least Squares
        # y = x * M  -> M = (x^T * x)^-1 * x^T * y
        # In NumPy: linalg.lstsq solves for the matrix that minimizes squared error
        matrix, residuals, rank, s = np.linalg.lstsq(x_data, y_data, rcond=None)
        
        # Transpose if necessary depending on your engine's multiplication order
        # Most VFX engines (Nuke/OCIO) use: [R, G, B] * Matrix
        result.matrix_3x3 = matrix.astype(np.float32)

        # --- New: Calculate Residual Delta E ---
        # Apply the matrix to all color patches to see how well it worked
        errors = []
        for p in result.patches:
            if p.index < 18: # Only check color patches
                # Apply matrix: [1x3] * [3x3] = [1x3]
                corrected_rgb = np.dot(p.observed_rgb, result.matrix_3x3)
                
                # Simple Euclidean Distance (Delta E approximation in Linear Space)
                # For high-accuracy, we would convert to Lab, but this is 
                # excellent for relative "Post-Matrix" validation.
                diff = corrected_rgb - p.target_rgb
                de = np.sqrt(np.sum(diff**2))
                errors.append(de)

        if errors:
            result.delta_e_mean = float(np.mean(errors))
            result.delta_e_max = float(np.max(errors))
        
        return result

    def perform_audit(self, audit_result: AuditResult) -> AuditResult:
        if not audit_result.patches: return audit_result

        # --- TASK 1: DNA VERIFICATION ---
        dna_valid = self.verify_dna(audit_result)
        # --------------------------------

        total_de = 0.0
        max_de = 0.0

        for patch in audit_result.patches:
            de = self.calculate_delta_e(patch.observed_rgb, patch.target_rgb)
            patch.delta_e = de 
            total_de += de
            max_de = max(max_de, de)

        count = len(audit_result.patches)
        audit_result.delta_e_mean = total_de / count if count > 0 else 0.0
        audit_result.delta_e_max = max_de
        # Audit passes ONLY if Delta E is low AND the DNA check passed
        audit_result.is_pass = (audit_result.delta_e_mean <= settings.tolerance_threshold) and dna_valid
        
        if not audit_result.timestamp:
            audit_result.timestamp = datetime.now().isoformat()

        return self.calculate_cdl_correction(audit_result)
    
    def verify_dna(self, result: AuditResult) -> bool:
        """
        Relational Integrity: Does the found grid behave like the template?
        Checks for descending luminance AND a minimum contrast threshold.
        """
        neutrals = result.get_neutral_patches()
        
        # If we find 0 patches, we have a locator failure.
        if len(neutrals) < 2: 
            result.ai_reasoning += " | CRITICAL: No neutral patches found."
            return False
        
        lums = [np.dot(p.observed_rgb, [0.2126, 0.7152, 0.0722]) for p in neutrals]
        
        # 1. Monotonicity (The Direction)
        is_monotonic = all(x >= (y - 0.005) for x, y in zip(lums, lums[1:]))
        
        # 2. Dynamic Range (The Magnitude)
        # A real Macbeth white-to-black ramp drops by roughly 2.0 stops or more.
        # We expect a minimum luminance spread of at least 0.1 in ACEScg.
        total_range = max(lums) - min(lums)
        is_dynamic = total_range > 0.1 

        if not is_monotonic:
            result.ai_reasoning += " | CRITICAL: DNA Failure (Non-linear ramp)."
            result.alignment_integrity = 0.0 
            return False

        if not is_dynamic:
            result.ai_reasoning += f" | CRITICAL: DNA Failure (No contrast - Flat Signal: {total_range:.3f})."
            result.alignment_integrity = 0.0
            return False
            
        return True
    
    def apply_visual_correction(self, result: AuditResult):
        """
        Prepares RGBs for the split-triangle view in UI/PDF.
        Uses the intent to decide which side gets the 'correction'.
        """
        for p in result.patches:
            # Determine the 'source' to be corrected
            if result.analysis_intent == "neutralize":
                # We want to see if the Plate can match the Reference
                src_rgb = p.observed_rgb
                ref_rgb = p.target_rgb
            else:
                # Match Grade: Can the Reference match the Plate?
                src_rgb = p.target_rgb
                ref_rgb = p.observed_rgb

            # Apply SOP + Matrix (Tier-dependent)
            corrected = self.color_engine.apply_math(src_rgb, result)
            
            # Store for the Exporter
            p.visual_src_rgb = corrected  # The 'Adjusted' side
            p.visual_ref_rgb = ref_rgb    # The 'Target' side