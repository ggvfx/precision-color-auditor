"""
Quality Assurance Reporting.

Generates technical CSV audit logs and human-readable TXT summaries.
Provides structured feedback on exposure drift and color accuracy.
"""

import csv
import numpy as np
from datetime import datetime
from fpdf import FPDF
import os
from core.models import AuditResult
from core.config import settings
from src.exporters.utils import get_system_metadata, format_corners
from src.core.templates import CHART_LIBRARY

class ReportGenerator:
    """Public API for all exporter tasks."""

    @staticmethod
    def write_csv(result: AuditResult, output_path: str):
        """
        Generates a flat CSV log of the audit results.
        Flattens the 3x3 matrix and SOP values for database ingestion.
        """
        sys_meta = get_system_metadata()

        header = [
            "timestamp", "hostname", "user", "version",
            "file_path", "camera_make", "camera_model",
            "template", "analysis_intent", 
            "input_space", "audit_space", "display_space",
            "integrity_score", "de_mean", "de_max",
            "slope_r", "slope_g", "slope_b",
            "offset_r", "offset_g", "offset_b",
            "power_r", "power_g", "power_b",
            "sat", "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
            "chart_corners_coord",
            "reasoning"
        ]

        # Flatten Matrix
        m = result.matrix_3x3.flatten() if result.matrix_3x3 is not None else [1,0,0,0,1,0,0,0,1]

        row = [
            result.timestamp, sys_meta["hostname"], sys_meta["user"], sys_meta["version"],
            result.file_path, result.camera_make, result.camera_model,
            result.template_name, result.analysis_intent,
            result.input_space or settings.default_input_space,
            result.audit_space or settings.default_audit_space,
            result.display_space or settings.default_display_space,
            f"{result.alignment_integrity:.4f}",
            f"{result.delta_e_mean:.6f}", f"{result.delta_e_max:.6f}",
            *result.slope, *result.offset, *result.power,
            f"{result.sat:.4f}", *m,
            format_corners(result.corners),
            result.ai_reasoning.strip(" | ")
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
            
        return output_path
    
    @staticmethod
    def write_pdf(result: AuditResult, output_path: str):
        """Generates the visual Audit Report."""
        pdf = AuditPDF()
        pdf.add_page()
        pdf.draw_summary_table(result)
        pdf.draw_patch_grid(result)
        pdf.draw_technical_data(result)
        pdf.output(output_path)
        return output_path

class AuditPDF(FPDF):
    """Internal helper to handle FPDF drawing operations."""
    
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Precision Color Audit Report", ln=True, align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def draw_summary_table(self, res):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "1. Audit Context", ln=True)
        
        self.set_font("Helvetica", "", 9)
        # Using the refactored 'target_space' and 'camera' metadata
        data = [
            ["File Path", res.file_path],
            ["Camera Info", f"{res.camera_make} {res.camera_model}"],
            ["Analysis Intent", res.analysis_intent.upper()],
            ["Input Space", res.input_space or settings.default_input_space],
            ["Audit Space", res.audit_space or settings.default_audit_space],
            ["Display Space", res.display_space or settings.default_display_space],
            ["Integrity Score", f"{res.alignment_integrity:.4f}"],
            ["Timestamp", str(res.timestamp)]
        ]
        
        for i, row in enumerate(data):
            self.set_fill_color(245, 245, 245)
            self.cell(40, 6, row[0], border=1, fill=True)
            
            # Check if it's the last row
            is_last = (i == len(data) - 1)
            
            # Use ln=True for all but the last row, or reset manually
            self.cell(150, 6, str(row[1]), border=1, ln=True)

        # Force a small gap after the table is fully drawn
        self.ln(2)

    def draw_patch_grid(self, res):
        self.set_font("Helvetica", "B", 12)
        
        # 1. Dynamic Header and Legend
        intent_label = "Neutralize" if res.analysis_intent == "neutralize" else "Match Grade"
        self.cell(0, 10, f"2. Visual Verification (Intent: {intent_label})", ln=True)
        
        # Key for the user
        self.set_font("Helvetica", "", 9)
        if res.analysis_intent == "neutralize":
            left_txt = "Left: Neutralized Plate (Corrected)"
            right_txt = "Right: Target Values (Reference)"
        else:
            left_txt = "Left: Match Grade (Corrected Target)"
            right_txt = "Right: Input Plate (Observed)"
            
        self.cell(0, 5, f"{left_txt}  |  {right_txt}", ln=True)
        self.ln(5)

        start_x, start_y = 20, self.get_y()
        size = 22

        template = CHART_LIBRARY.get(res.template_name)
        
        if template and template.topology == "grid":
            cols, rows = template.grid
        else:
            cols = 5 
            rows = (len(res.patches) // 5) + 1
        
        def to_8bit(val):
            return int(np.clip(val ** (1/2.2), 0, 1) * 255)

        for i, p in enumerate(res.patches):
            col = i % 6
            row = i // 6
            x = start_x + (col * (size + 4))
            y = start_y + (row * (size + 2))

            # --- DYNAMIC TRIANGLE ASSIGNMENT ---
            # Based on the logic in Auditor.apply_visual_correction
            # visual_src_rgb is the 'Corrected' side
            # visual_ref_rgb is the 'Reference/Goal' side
            
            fill_left = getattr(p, 'visual_src_rgb', None) if getattr(p, 'visual_src_rgb', None) is not None else p.observed_rgb
            fill_right = getattr(p, 'visual_ref_rgb', None) if getattr(p, 'visual_ref_rgb', None) is not None else p.target_rgb

            # Ultimate safety fallback if even observed/target are None
            if fill_left is None: fill_left = np.array([0,0,0])
            if fill_right is None: fill_right = np.array([0,0,0])

            # Draw Left Triangle (Corrected Side)
            self.set_fill_color(*[to_8bit(c) for c in fill_left])
            self.polygon([(x, y), (x + size, y), (x, y + size)], fill=True)
            
            # Draw Right Triangle (Reference Side)
            self.set_fill_color(*[to_8bit(c) for c in fill_right])
            self.polygon([(x + size, y), (x + size, y + size), (x, y + size)], fill=True)
            
            self.set_draw_color(100, 100, 100)
            self.rect(x, y, size, size)
        
        cols = 6
        rows = (len(res.patches) + (cols - 1)) // cols
        grid_height = rows * (size + 2)
        self.set_y(start_y + grid_height + 10)

    def draw_technical_data(self, res):
        if self.get_y() > 250:
            self.add_page()
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "3. Mathematical Results", ln=True)
        
        # Use a single Multi-cell for the whole block to keep it contained
        m = res.matrix_3x3
        tech_text = (
            f"ASC-CDL:\n"
            f"  SLOPE:  {res.slope[0]:.4f} {res.slope[1]:.4f} {res.slope[2]:.4f}\n"
            f"  OFFSET: {res.offset[0]:.4f} {res.offset[1]:.4f} {res.offset[2]:.4f}\n"
            f"  SAT:    {res.sat:.4f}\n\n"
            f"COLOR MATRIX:\n"
            f"  [{m[0][0]:.3f}, {m[0][1]:.3f}, {m[0][2]:.3f}]\n"
            f"  [{m[1][0]:.3f}, {m[1][1]:.3f}, {m[1][2]:.3f}]\n"
            f"  [{m[2][0]:.3f}, {m[2][1]:.3f}, {m[2][2]:.3f}]"
        )
        self.set_font("Courier", "", 9)
        self.multi_cell(0, 5, tech_text, border=1, fill=False)