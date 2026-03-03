# Precision Color Auditor

**Precision Color Auditor** is a professional, standalone calibration instrument designed for Digital Imaging Professionals, VFX Supervisors, and Photographers. By leveraging local computer vision models and template-driven geometric rectification, the tool automates the "ground truth" verification of digital images, identifying color drift and exposure errors while providing deterministic neutralization via non-destructive sidecar metadata.

## Project Status
🚦 **Project Status:** Alpha (Macro & Micro Pipelines Validated)
The project has established a robust, local AI inference pipeline using **Florence-2** for localization and a **Template-Driven Topology** engine for precise patch sampling.

**Current Milestone:** Implementing the **OCIO Audit Engine** to calculate Delta E 2000 errors and ASC-CDL neutralization values.

## Strategic Roadmap

### Phase 1: Ingestion & AI Localization (Complete)
* **Unified Ingest Engine:** Core logic for loading and linearizing high-bit-depth formats (EXR, RAW, DPX).
* **AI Chart Localization:** Integrated **Florence-2** for identifying Macbeth and Grayscale chart regions.
* **Coordinate Mapping:** Normalization of AI-detected regions to full-resolution image coordinates.

### Phase 2: Rectification & Sampling (Complete)
* **Perspective Rectification:** OpenCV routines warp detected charts into a flat, standardized coordinate space based on template aspect ratios.
* **Template-Driven Topology:** Dynamic support for various charts (Macbeth 24, Kodak Gray Card) via a centralized `ChartTemplate` library.
* **Deterministic Grid Sampling:** High-precision mean-pixel extraction using configurable inset margins to avoid bezel contamination.
* **Orientation Intelligence:** Logic to detect flipped or rotated charts via luminance gradient analysis of the grayscale ramp.

### Phase 3: The Auditor & OCIO (Current)
* **OCIO Integration:** Integration of **OpenColorIO v2** to transform sampled data into scene-linear "Audit Space."
* **Signal Analysis:** Leveraging `colour-science` to quantify color accuracy using **Delta E 2000** metrics.
* **CDL Generation:** Calculating required **Slope, Offset, and Power** (ASC-CDL) to align observed data with mathematical ground truth.

### Phase 4: Professional Exports & Reporting (Upcoming)
* **Neutralization Exporters:** Industry-standard **ASC-CDL** (.cdl, .cc) and **3D LUT** (.cube) generation.
* **Human-in-the-Loop UI:** PySide6 interface for manual corner refinement and "QC Crop" verification.
* **Automated QA Reporting:** Generation of CSV and PDF reports for production hand-offs and VFX plate delivery.

## 🚀 Overview
The **Precision Color Auditor** eliminates "color drift" at the source. By moving beyond visual "eye-balling" and into deterministic signal analysis, it empowers color-critical professionals to ensure every plate matches a mathematical ideal before production begins.

## 🚀 Philosophy
The **Precision Color Auditor** follows a **"Precision-First"** and **"Non-Destructive"** workflow:
1.  **Visual Intelligence:** Uses local transformer-based models to find charts in complex scenes without cloud dependencies.
2.  **Geometric Truth:** Warps charts into a flat, rectified space to ensure pixel-perfect sampling regardless of camera angle.
3.  **Dynamic Templates:** Supports an extensible library of physical charts via a standardized dataclass architecture.
4.  **Metadata-Only Fixes:** Identifies errors without "baking" them in, providing sidecar CDLs to maintain original pixel integrity.

## 🛠️ Key Features
* **Template-Driven Architecture:** Easily add support for new charts by defining their grid or anchored topology.
* **Geometric Rectification:** Warps skewed or angled charts into a flat plane for consistent sampling.
* **QC Output Pipeline:** Generates rectified "Verification Crops" with sample-point overlays for visual audit.
* **OCIO-Native Workflow:** Powered by OpenColorIO v2 for color-critical accuracy.
* **Delta E Signal Analysis:** Quantifies color errors using industry-standard dE2000 formulas.
* **Multi-Format Precision:** Native support for Cinema (EXR/DPX) and Photography (RAW) pipelines.

## 🛠️ Technical Stack
* **Language:** Python 3.11+
* **Intelligence:** Florence-2 (Local Vision Model)
* **Image Processing:** OpenCV (Perspective Warping & Grid Mapping)
* **Color Science:** OpenColorIO v2.x, `colour-science`
* **Image Ingest:** OpenImageIO, Rawpy, OpenCV
* **UI Framework:** PySide6 (Qt)

## 📂 Project Structure
* `src/core/`: Settings, template library, color models, and auditor logic.
* `src/ai/`: Florence-2 integration, locator (Macro pass), and topology (Micro pass).
* `src/exporters/`: CDL, LUT, and Report generation modules.
* `src/ui/`: PySide6 components for the calibration interface.