"""
Metadata Fix Generation.

Writes industry-standard ASC-CDL (.cdl, .cc) sidecar files.
Translates neutralization math into Slope, Offset, and Power values.
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from core.models import AuditResult

class CDLWriter:
    @staticmethod
    def write(result: AuditResult, output_path: str):
        """
        Writes ASC-CDL XML. 
        Supports template-specific logic (e.g., 'gray_card' limits to Slope).
        """
        # 1. Setup Root and Corrections
        root = ET.Element("ColorDecisionList", xmlns="urn:ASC:CDL:v1.01")
        decision = ET.SubElement(root, "ColorDecision")
        cc = ET.SubElement(decision, "ColorCorrection", id=result.template_name)
        sop = ET.SubElement(cc, "SOPOperation")

        # 2. Extract Values based on Template capabilities
        # For gray_card, we lock Offset to 0 and Power to 1
        is_slope_only = "Slope Only" in str(result.template_name) # Or check config
        
        slope_vals = result.slope
        offset_vals = result.offset if not is_slope_only else [0.0, 0.0, 0.0]
        power_vals = result.power if not is_slope_only else [1.0, 1.0, 1.0]

        def fmt(vals): return f"{vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f}"

        # 3. Populate SOP
        ET.SubElement(sop, "Slope").text = fmt(slope_vals)
        ET.SubElement(sop, "Offset").text = fmt(offset_vals)
        ET.SubElement(sop_node := sop, "Power").text = fmt(power_vals)
        
        # 4. Saturation
        sat_node = ET.SubElement(cc, "SatOperation")
        ET.SubElement(sat_node, "Saturation").text = f"{result.sat:.6f}"
        
        # 5. Metadata Description
        desc = ET.SubElement(cc, "Description")
        desc.text = (f"PCA Audit | Intent: {result.analysis_intent} | "
                     f"Integrity: {result.alignment_integrity:.2f}")

        # Pretty Print
        xml_str = ET.tostring(root, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        with open(output_path, "w") as f:
            f.write(pretty_xml)
            
        return output_path