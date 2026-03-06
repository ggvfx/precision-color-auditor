"""
Session Module - The Conductor
---------------------------------------------------
Responsibility: Centralized State, Batch Management, and Persistence.

Features:
- PySide6 Threading: Process images in the background without freezing the UI.
- Recursive Search: Find images nested deep in directory structures.
- Intelligent Filtering: Include/Exclude based on folder names or filenames.
- Sidecar Persistence: Save/Load .pca.json files next to source images.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

from PySide6.QtCore import QObject, Signal, QThread, Slot
from .models import AuditResult, ColorPatch

class AuditWorker(QObject):
    """
    Background worker for the SessionManager. 
    Handles the heavy lifting of AI detection and sampling.
    """
    image_done = Signal(AuditResult)  # One image finished
    batch_done = Signal()           # Entire list finished
    error = Signal(str)             # Error reporting

    def __init__(self, sampler, file_list: List[str]):
        super().__init__()
        self.sampler = sampler
        self.file_list = file_list
        self._abort = False

    @Slot()
    def run(self):
        for file_path in self.file_list:
            if self._abort:
                break
            
            try:
                # Note: In the final app, the SessionManager/UI will provide 
                # the actual image buffers to the sampler here.
                # result = self.sampler.sample_all(display_buf, audit_buf, file_path)
                # self.image_done.emit(result)
                pass
            except Exception as e:
                self.error.emit(f"Failed {file_path}: {str(e)}")
        
        self.batch_done.emit()

    def stop(self):
        self._abort = True


class SessionManager(QObject):
    """
    The main 'Brain' of the application.
    Connects the UI to the AI/Color Logic.
    """
    # Signals for UI updates
    progress_changed = Signal(int, int) # current, total
    image_updated = Signal(str)        # file_path updated in results
    batch_finished = Signal()

    def __init__(self, engine=None, sampler=None):
        super().__init__()
        self.engine = engine
        self.sampler = sampler
        
        # The Source of Truth: Mapping of file_path -> AuditResult
        self.results: Dict[str, AuditResult] = {}
        self.image_extensions = {'.exr', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        
        # Threading handles
        self._thread: Optional[QThread] = None
        self._worker: Optional[AuditWorker] = None

    # --- 1. FILE DISCOVERY & FILTERING ---

    def load_work_area(self, root_path: str, recursive: bool = True, 
                       folder_filters: List[str] = None, 
                       file_filters: List[str] = None) -> List[str]:
        """
        Scans for images and automatically restores data from Sidecar JSONs.
        """
        root = Path(root_path)
        pattern = "**/*" if recursive else "*"
        found_files = []

        # Normalize filters for case-insensitive matching
        f_filters = [f.lower() for f in (folder_filters or [])]
        n_filters = [n.lower() for n in (file_filters or [])]

        for file in root.expanduser().rglob(pattern):
            if file.suffix.lower() not in self.image_extensions:
                continue

            # Check Folder Filters (e.g., 'charts', 'macbeth')
            if f_filters and not any(f in file.parent.name.lower() for f in f_filters):
                continue

            # Check Filename Filters (e.g., '_REF', 'chart_')
            if n_filters and not any(n in file.name.lower() for n in n_filters):
                continue

            file_str = str(file.resolve())
            found_files.append(file_str)

            # Check for existing .pca.json next to the image
            sidecar = file.with_suffix(file.suffix + ".pca.json")
            if sidecar.exists():
                self._deserialize_sidecar(file_str, sidecar)
        
        return found_files

    # --- 2. PERSISTENCE (SIDECARS) ---

    def _deserialize_sidecar(self, image_path: str, sidecar_path: Path):
        """Reconstructs AuditResult from a JSON file on disk."""
        try:
            with open(sidecar_path, 'r') as f:
                data = json.load(f)
            
            patches = [
                ColorPatch(
                    name=p['name'],
                    local_center=tuple(p['local_center']),
                    delta_e=p['delta_e'],
                    index=i,
                    observed_rgb=np.zeros(3), # Placeholder until re-sampled
                    target_rgb=np.zeros(3)
                ) for i, p in enumerate(data.get('patches', []))
            ]

            self.results[image_path] = AuditResult(
                file_path=image_path,
                template_name=data.get('template', 'unknown'),
                corners=np.array(data['corners']) if data['corners'] else None,
                is_pass=data.get('is_pass', False),
                patches=patches,
                ai_reasoning="Loaded from Sidecar"
            )
        except Exception as e:
            print(f"[ERROR] Could not parse sidecar for {image_path}: {e}")

    def save_sidecar(self, file_path: str):
        """Saves AuditResult to a JSON file next to the original image."""
        if file_path not in self.results:
            return

        res = self.results[file_path]
        img_path = Path(file_path)
        sidecar_path = img_path.with_suffix(img_path.suffix + ".pca.json")

        payload = {
            "template": res.template_name,
            "is_pass": res.is_pass,
            "corners": res.corners.tolist() if res.corners is not None else None,
            "patches": [
                {
                    "name": p.name,
                    "delta_e": round(float(p.delta_e), 4),
                    "local_center": p.local_center
                } for p in res.patches
            ]
        }

        with open(sidecar_path, 'w') as f:
            json.dump(payload, f, indent=4)

    # --- 3. BATCH PROCESSING ---

    def run_batch(self, file_list: List[str]):
        """Initializes and starts the background worker thread."""
        if self._thread and self._thread.isRunning():
            return # Avoid double-processing

        self._thread = QThread()
        self._worker = AuditWorker(self.sampler, file_list)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.image_done.connect(self._on_worker_image_done)
        self._worker.batch_done.connect(self._thread.quit)
        self._worker.batch_done.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: self.batch_finished.emit())

        self._thread.start()

    def _on_worker_image_done(self, result: AuditResult):
        """Internal callback for the background thread."""
        self.results[result.file_path] = result
        self.save_sidecar(result.file_path)
        self.image_updated.emit(result.file_path)