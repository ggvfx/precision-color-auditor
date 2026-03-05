"""
Precision Color Auditor - Signal Analysis & Neutralization Logic
Standardized to template-based targets and ACEScg workflow.
"""

import numpy as np
import colour
from core.models import ColorPatch, AuditResult
from core.config import settings
from datetime import datetime

class Auditor:
    def __init__(self):
        pass

    def calculate_delta_e(self, observed_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
        if np.all(target_rgb <= 0): return 0.0
        obs_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(observed_rgb, 'ACEScg'))
        targ_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(target_rgb, 'ACEScg'))
        return float(colour.delta_E(obs_lab, targ_lab, method='CIE 2000'))

    def calculate_cdl_correction(self, audit_result: AuditResult) -> AuditResult:
        """
        Dispatches to the correct math based on the template's analysis_mode.
        """
        template = settings.get_current_template()
        mode = getattr(template, 'analysis_mode', 'gain')

        # DISPATCHER DEBUG
        print(f"--- DISPATCHER DEBUG ---")
        print(f"Template Name: {template.name}")
        print(f"Analysis Mode: {mode}")
        print(f"Patch Count: {len(audit_result.patches)}")

        # Map the patches for easy lookup
        # In calculate_cdl_correction
        patch_map = {}
        for p in audit_result.patches:
            clean_name = p.name.replace("Patch_", "").strip().lower()
            patch_map[clean_name] = p
            patch_map[str(p.index)] = p # Allows lookup by '18' or 18

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

    def _solve_gain(self, result: AuditResult) -> AuditResult:
        """TIER 1: Single patch balance (Slope only, Offset 0)."""
        if not result.patches: return result
        obs = result.patches[0].observed_rgb
        targ = result.patches[0].target_rgb
        
        result.slope = np.clip(targ / (obs + 1e-6), 0.0, 4.0).astype(np.float32)
        result.offset = np.zeros(3, dtype=np.float32)
        return result

    def _solve_anchors(self, result: AuditResult, patch_map: dict) -> AuditResult:
        """TIER 2: Anchor Set. Linear regression across all identified neutrals."""
        template = settings.get_current_template()
        neutral_names = template.neutral_indices

        # DIAGNOSTIC PRINTS
        print(f"DEBUG: Active Template: {template.name}")
        print(f"DEBUG: Looking for Neutrals: {neutral_names}")
        print(f"DEBUG: Keys in Patch Map: {list(patch_map.keys())}")
        
        obs_list = []
        targ_list = []

        for name in neutral_names:
            # Normalize the search key to match our map
            search_key = str(name).strip().lower()
            p = patch_map.get(search_key)
            
            if p:
                obs_list.append(p.observed_rgb)
                targ_list.append(p.target_rgb)
            else:
                print(f"DEBUG: Failed to find patch for neutral key: {search_key}")

        if len(obs_list) < 2: 
            return result # Not enough data to draw a line

        obs = np.array(obs_list)
        targ = np.array(targ_list)

        slopes, offsets = [], []
        for i in range(3):
            m, c = np.polyfit(obs[:, i], targ[:, i], 1)
            slopes.append(m)
            offsets.append(c)

        result.slope = np.clip(slopes, 0.0, 4.0).astype(np.float32)
        result.offset = np.clip(offsets, -1.0, 1.0).astype(np.float32)
        return result

    def _solve_ramp(self, result: AuditResult, index_map: dict, neutral_indices: list) -> AuditResult:
        """TIER 3/4: Macbeth/Ramp Style. Regression fit through multiple points."""
        obs_list = []
        targ_list = []

        for idx in neutral_indices:
            p = index_map.get(idx)
            if p:
                obs_list.append(p.observed_rgb)
                targ_list.append(p.target_rgb)

        if len(obs_list) < 2: return result

        obs = np.array(obs_list)
        targ = np.array(targ_list)

        # Use Linear Regression (polyfit) for best-fit Slope/Offset across all ramp points
        slopes, offsets = [], []
        for i in range(3):
            m, c = np.polyfit(obs[:, i], targ[:, i], 1)
            slopes.append(m)
            offsets.append(c)

        result.slope = np.clip(slopes, 0.0, 4.0).astype(np.float32)
        result.offset = np.clip(offsets, -1.0, 1.0).astype(np.float32)
        
        # Note: 'color' mode (Macbeth) uses this CDL, but can be extended 
        # later to generate a 3D LUT via the color patches.
        return result
    
    def _solve_color(self, result: AuditResult, index_map: dict, neutral_indices: list) -> AuditResult:
        """TIER 4: Macbeth. Neutralization CDL + Future Matrix Profiling."""
        # 1. First, neutralize using the ramp (Row 4 of the Macbeth)
        result = self._solve_ramp(result, index_map, neutral_indices)
        
        # 2. PLACEHOLDER: Full Color Matrix
        # Compare the 18 color patches (observed vs target) to solve for a 3x3 matrix.
        # result.matrix_3x3 = self._solve_3x3_matrix(result)
        
        return result

    def perform_audit(self, audit_result: AuditResult) -> AuditResult:
        if not audit_result.patches: return audit_result

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
        audit_result.is_pass = audit_result.delta_e_mean <= settings.tolerance_threshold
        
        if not audit_result.timestamp:
            audit_result.timestamp = datetime.now().isoformat()

        return self.calculate_cdl_correction(audit_result)