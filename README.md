# Precision Color Auditor

**Precision Color Auditor** is a professional, standalone calibration instrument designed for Digital Imaging Professionals, VFX Supervisors, and Photographers. By leveraging local computer vision models and deterministic geometric rectification, the tool automates the "ground truth" verification of digital images, identifying color drift and exposure errors while providing deterministic neutralization via non-destructive sidecar metadata.

## Project Status
🚦 **Project Status:** Alpha (Macro Pipeline Validated)
The project has established a robust, local AI inference pipeline using Florence-2 for initial localization. We have shifted from "Zero-Shot" patch detection to a **Geometric Rectification** strategy to ensure 100% coordinate accuracy for professional auditing.

**Current Milestone:** Implementing **Geometric Rectification** and **Grid-Based Sampling** to extract mean-pixel data from perspective-warped chart crops.

## Strategic Roadmap

### Phase 1: Ingestion & AI Localization (Complete)
* **Unified Ingest Engine:** Core logic for loading and linearizing high-bit-depth formats (EXR, RAW, DPX).
* **AI Chart Localization:** Integrated **Florence-2** for identifying Macbeth and Grayscale chart regions and primary anchor corners.
* **Coordinate Mapping:** Normalization of AI-detected regions to full-resolution image coordinates.

### Phase 2: Rectification & Sampling (Current)
* **Perspective Rectification:** (New) Build OpenCV routines to warp detected charts into a flat, normalized coordinate space ("QC Crop").
* **Grid-Based Sampling:** Mathematical 4x6 grid extraction for Macbeth patches, replacing erratic AI point detection with deterministic geometry.
* **OCIO Integration:** Build the transformation engine using **OpenColorIO v2** to move sampled data into a scene-linear "Audit Space."
* **The Auditor:** Implement `colour-science` math to calculate Delta E 2000 errors and determine necessary Slope/Offset/Power adjustments.

### Phase 3: Professional Exports & Reporting (Upcoming)
* **Neutralization Fixes:** Finalize exporters for industry-standard **ASC-CDL** (.cdl, .cc) and **3D LUT** (.cube).
* **Human-in-the-Loop QC:** Integrated UI for manual corner refinement and "QC Crop" verification.
* **Batch & QA Reporting:** Automated "Quality Assurance" reports in CSV and human-readable formats for production hand-offs.

## 🚀 Overview
The **Precision Color Auditor** eliminates "color drift" at the source. By moving beyond visual "eye-balling" and into deterministic signal analysis, it empowers color-critical professionals to ensure every plate matches a mathematical ideal before production begins.

The tool follows a **"Precision-First"** and **"Non-Destructive"** philosophy:
1.  **Visual Intelligence:** Uses local transformer-based models to find charts and anchor corners in any orientation or lighting condition.
2.  **Geometric Truth:** Warps charts into a flat, rectified space to ensure pixel-perfect sampling of internal patch structures.
3.  **Deterministic Audit:** All audits are performed in scene-linear space using industry-standard color libraries.
4.  **Metadata-Only Fixes:** Identifies errors without "baking" them in, providing sidecar CDLs to maintain pixel integrity.

## 🛠️ Key Features
* **Geometric Rectification:** Warps skewed or angled charts into a flat plane for consistent sampling.
* **QC Output Pipeline:** Automatically generates rectified "Verification Crops" so users can audit the AI's alignment.
* **OCIO-Native Workflow:** Powered by the same engine used in major VFX houses (OpenColorIO v2).
* **Delta E Signal Analysis:** Quantifies color errors using Delta E 2000 metrics to measure drift from neutral.
* **Automated Neutralization:** Calculates required **Slope, Offset, and Power** (ASC-CDL) for correction.
* **Multi-Format Precision:** Bridges the gap between RAW (Photography) and EXR/DPX (Cinema) pipelines.

## 🛠️ Technical Stack
* **Language:** Python 3.11+
* **UI Framework:** PySide6 (Qt)
* **Intelligence:** Florence-2 (Local Vision Model)
* **Image Processing:** OpenCV (Perspective Warping & Grid Mapping)
* **Color Math:** OpenColorIO (OCIO) v2.x, colour-science
* **Image Ingest:** OpenImageIO, Rawpy, OpenCV

## 📂 Project Structure
* `src/core/`: Ingest, models, color engine, and auditor logic.
* `src/ai/`: Florence-2 integration, locator (Macro), and topology (Micro).
* `src/exporters/`: CDL, LUT, and Report generation modules.
* `src/ui/`: PySide6 components for the calibration interface.