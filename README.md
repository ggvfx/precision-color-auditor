# Precision Color Auditor

**Precision Color Auditor** is a professional, standalone calibration instrument designed for Digital Imaging Professionals, VFX Supervisors, and Photographers. By leveraging local computer vision models and industry-standard color science, the tool automates the "ground truth" verification of digital images, identifying color drift and exposure errors while providing deterministic neutralization via non-destructive sidecar metadata.

## Project Status
🚦 **Project Status:** Pre-Alpha (Architecture & Environment Setup)
The project is currently in the initial setup and architectural planning phase. We have defined the Master Brief, established the technical stack, and initialized the development environment.

**Current Milestone:** Initializing the **Unified Ingest Layer** to verify multi-format support (EXR, RAW, DPX) and establishing the **Florence-2** local inference pipeline.

## Strategic Roadmap

### Phase 1: Ingestion & AI Detection (Upcoming)
* **Unified Ingest Engine:** Develop the core logic to load and linearize EXR, DPX, and RAW formats using OpenImageIO and Rawpy.
* **AI Chart Localization:** Implement **Florence-2** local inference to automatically detect and provide coordinates for Macbeth/Greyscale charts.
* **Sampling Logic:** Build the OpenCV routines for mean-pixel sampling of the 6 neutral patches.

### Phase 2: Color Engine & Auditor Logic
* **OCIO Integration:** Build the transformation engine using **OpenColorIO v2** to move sampled data into a scene-linear "Audit Space."
* **The Auditor:** Implement `colour-science` math to calculate Delta E errors and determine necessary Slope/Offset/Power adjustments.
* **Validation UI:** Create the initial PySide6 interface for real-time signal probes and tolerance thresholding.

### Phase 3: Professional Exports & Reporting
* **Neutralization Fixes:** Finalize exporters for industry-standard **ASC-CDL** (.cdl, .cc) and **3D LUT** (.cube) formats.
* **Batch & QA Reporting:** Engineering the automated "Quality Assurance" reports in CSV and human-readable text formats.

## 🚀 Overview
The **Precision Color Auditor** is a specialized utility designed to eliminate "color drift" and exposure inconsistency at the source. By moving beyond visual "eye-balling" and into deterministic signal analysis, it empowers color-critical professionals to ensure every plate or photograph matches a mathematical ideal before it enters a high-end production pipeline.

The tool follows a **"Precision-First"** and **"Non-Destructive"** philosophy:
1.  **Deterministic Truth:** All audits are performed in scene-linear space using the industry’s most respected color libraries.
2.  **Metadata-Only Fixes:** The tool identifies errors but never "bakes" them in, providing sidecar CDLs to maintain the integrity of the original pixels.

## 🛠️ Key Features
* **AI-Powered ROI Detection:** Automates the tedious task of finding and cropping color charts, regardless of the camera's orientation or distance.
* **OCIO-Native Workflow:** Uses the exact same color engine as major VFX and DI houses, ensuring the "Auditor's" math matches the "Artist's" view.
* **Delta E Signal Analysis:** Quantifies color errors using industry-standard Delta E 2000 metrics to tell you exactly how far a shot has drifted from neutral.
* **Automated Neutralization:** Calculates the necessary **Slope, Offset, and Power** (ASC-CDL) to correct for color temperature and exposure mismatches.
* **Multi-Format Precision:** Bridges the gap between Photography (RAW/DNG) and Cinema (EXR/DPX) color pipelines.
* **Audit Logging & Reporting:** Outputs comprehensive **CSV and TXT reports** for quality assurance and production hand-offs.

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