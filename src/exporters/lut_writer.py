"""
3D LUT File Generation.

Writes industry-standard .cube files for downstream neutralization fixes.
Encapsulates mathematical transforms into baked spatial look-up tables.
"""
import numpy as np
from core.models import AuditResult

class LUTWriter:
    @staticmethod
    def write_3d_cube(result: AuditResult, output_path: str, size: int = 33):
        """
        Generates a 3D LUT (.cube) combining the CDL and the 3x3 Matrix.
        Standard size is 33 (33^3 = 35,937 lines).
        """
        # Header requirements for OCIO/Nuke
        header = [
            f'# Precision Color Auditor Extraction',
            f'# Intent: {result.analysis_intent}',
            f'TITLE "{result.template_name}"',
            f'LUT_3D_SIZE {size}',
            'DOMAIN_MIN 0.0 0.0 0.0',
            'DOMAIN_MAX 1.0 1.0 1.0',
            ''
        ]

        # 1. Generate the 3D Lattice
        # np.linspace creates 'size' steps from 0 to 1
        vals = np.linspace(0.0, 1.0, size).astype(np.float32)
        
        # meshgrid with 'ij' indexing creates the grid
        # To match .cube standard (R fastest), we use this order:
        # The loop order in the file must be:
        # for b in range(size): for g in range(size): for r in range(size):
        blue, green, red = np.meshgrid(vals, vals, vals, indexing='ij')
        
        # Stack into a list of RGB triplets
        pixels = np.stack([red, green, blue], axis=-1).reshape(-1, 3)

        # 2. Apply the Math
        # Apply CDL (SOP): (pixels * slope) + offset
        # Note: We'll skip Power/Saturation for this baseline to keep it linear
        transformed = (pixels * result.slope) + result.offset
        
        # Apply 3x3 Matrix from Task 3
        if result.matrix_3x3 is not None:
            transformed = np.dot(transformed, result.matrix_3x3)

        # 3. Write to File
        with open(output_path, "w") as f:
            f.write("\n".join(header) + "\n")
            # Format: R G B (0.000000 0.000000 0.000000)
            np.savetxt(f, transformed, fmt='%.6f')
                
        return output_path