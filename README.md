# Precision Color Auditor

**Precision Color Auditor** is a professional, standalone calibration instrument designed for Digital Imaging Professionals, VFX Supervisors, and Photographers. By leveraging local computer vision models and industry-standard color science, the tool automates the "ground truth" verification of digital images, identifying color drift and exposure errors while providing deterministic neutralization via non-destructive sidecar metadata.

## Project Status
🚦 **Project Status:** Alpha (Core AI Pipeline Validated)
The project has successfully moved past architectural planning. There is established a robust, local AI inference pipeline using Florence-2, capable of dynamic chart discovery without the need for rigid geometric templates.

**Current Milestone:** Implementing **Sub-Region Sampling** to extract mean-pixel data from AI-detected ROIs and integrating **OpenColorIO v2** for scene-linear math.

## Strategic Roadmap

### Phase 1: Ingestion & AI Discovery (Complete)
* **Unified Ingest Engine:** Core logic for loading and linearizing high-bit-depth formats (EXR, RAW, DPX).
* **AI Visual Discovery:** Integrated **Florence-2** for zero-shot detection of Macbeth and Grayscale charts, replacing legacy template-matching dependencies.
* **Coordinate Mapping:** Normalization of AI-detected bounding boxes to full-resolution image coordinates.

### Phase 2: Color Engine & Sampling (Current)
* **Precision Sampling:** Build OpenCV routines for mean-pixel sampling of internal patch structures within AI-detected ROIs.
* **OCIO Integration:** Build the transformation engine using **OpenColorIO v2** to move sampled data into a scene-linear "Audit Space."
* **The Auditor:** Implement `colour-science` math to calculate Delta E errors and determine necessary Slope/Offset/Power adjustments.

### Phase 3: Professional Exports & Reporting (Upcoming)
* **Neutralization Fixes:** Finalize exporters for industry-standard **ASC-CDL** (.cdl, .cc) and **3D LUT** (.cube).
* **Batch & QA Reporting:** Automated "Quality Assurance" reports in CSV and human-readable formats for production hand-offs.

## 🚀 Overview
The **Precision Color Auditor** eliminates "color drift" at the source. By moving beyond visual "eye-balling" and into deterministic signal analysis, it empowers color-critical professionals to ensure every plate matches a mathematical ideal before production begins.

The tool follows a **"Precision-First"** and **"Non-Destructive"** philosophy:
1.  **Visual Intelligence:** Uses local transformer-based models to "see" charts in any orientation, lighting condition, or focal length.
2.  **Deterministic Truth:** All audits are performed in scene-linear space using industry-standard color libraries.
3.  **Metadata-Only Fixes:** Identifies errors without "baking" them in, providing sidecar CDLs to maintain pixel integrity.

## 🛠️ Key Features
* **AI-Powered Visual Discovery:** Automates chart localization regardless of camera distance, rotation, or partial occlusion.
* **OCIO-Native Workflow:** Powered by the same engine used in major VFX houses (OpenColorIO v2).
* **Delta E Signal Analysis:** Quantifies color errors using Delta E 2000 metrics to measure drift from neutral.
* **Automated Neutralization:** Calculates required **Slope, Offset, and Power** (ASC-CDL) for correction.
* **Multi-Format Precision:** Bridges the gap between RAW (Photography) and EXR/DPX (Cinema) pipelines.

## 🛠️ Technical Stack
* **Language:** Python 3.11+
* **UI Framework:** PySide6 (Qt)
* **Intelligence:** Florence-2 (Local Vision Model)
* **Color Math:** OpenColorIO (OCIO) v2.x, colour-science
* **Image Ingest:** OpenImageIO, Rawpy, OpenCV

## 📂 Project Structure
* `src/core/`: Ingestion, color engine, and auditor logic.
* `src/ai/`: Florence-2 integration and chart detection.
* `src/exporters/`: CDL, LUT, and Report generation modules.
* `src/ui/`: PySide6 components for the calibration interface.