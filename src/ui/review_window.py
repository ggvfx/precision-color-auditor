import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTableWidget, QTableWidgetItem, QPushButton, 
                               QGroupBox, QCheckBox, QLabel, QFrame, QHeaderView,
                               QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
                               QFileDialog, QApplication, QStyledItemDelegate, QLineEdit)
from PySide6.QtCore import Qt, QSize, QPoint, QRectF, QRect
from PySide6.QtGui import QColor, QPainter, QPolygon, QPolygonF, QCursor, QImage, QPixmap
import qtawesome as qta

# --- PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))

from ui.widgets import (
    create_ocio_combo, 
    SampledChartDelegate, 
    TrianglePatchDelegate, 
    ChartMagnifier, 
    GridMagnifier
)
from core.color_engine import ColorEngine
from core.models import AuditStatus

class ReviewWindow(QMainWindow):
    def __init__(self, session_manager):
        super().__init__()
        self.session = session_manager
        self.color_engine = getattr(session_manager, 'color_engine', None) 
        if not self.color_engine and hasattr(session_manager, 'sampler'):
            self.color_engine = session_manager.sampler.color_engine
        self.setWindowTitle("Audit Review & Export")
        self.setMinimumSize(1800, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self._setup_table()
        self._setup_review_sidebar()

        self.table.setMouseTracking(True)
        self.magnifier = GridMagnifier()
        self.chart_magnifier = ChartMagnifier()
        
        self.table.cellEntered.connect(self._handle_hover)
        self.refresh_table()

    def _setup_table(self):
        self.table = QTableWidget(0, 11)
        headers = [
            "Sampled Chart", "Filename", "Camera Info", "Format", 
            "Resolution", "Input Space", "Intent", 
            "Visual Check", "Integrity", "Status", "Actions"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Apply both delegates
        self.table.setItemDelegateForColumn(0, SampledChartDelegate(self.table))
        self.table.setItemDelegateForColumn(7, TrianglePatchDelegate(self.table))
        
        header = self.table.horizontalHeader()
        self.table.setColumnWidth(12, 50)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(120)
        self.layout.addWidget(self.table, stretch=1)

    def refresh_table(self):
        self.table.setRowCount(0)
        for path, result in self.session.results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Store result in Col 0 for the delegate
            rect_item = QTableWidgetItem()
            rect_item.setData(Qt.UserRole, result)
            self.table.setItem(row, 0, rect_item)

            filename_item = QTableWidgetItem(result.file_path.split("/")[-1])
            filename_item.setData(Qt.UserRole, result.file_path) 
            self.table.setItem(row, 1, filename_item)

            self.table.setItem(row, 2, QTableWidgetItem(f"{result.camera_make} {result.camera_model}"))
            self.table.setItem(row, 3, QTableWidgetItem(result.file_path.split(".")[-1].upper()))
            self.table.setItem(row, 4, QTableWidgetItem(f"{getattr(result, 'width', 0)} x {getattr(result, 'height', 0)}"))
            # --- COLUMN 5: INPUT SPACE DROPDOWN ---
            combo = self._create_input_space_combo(row, result.input_space or "Default")
            self.table.setCellWidget(row, 5, combo)
            self.table.setItem(row, 6, QTableWidgetItem(result.analysis_intent.upper()))
            self.table.setItem(row, 8, QTableWidgetItem(f"{result.alignment_integrity:.4f}"))

            status_item = QTableWidgetItem()
            self._update_status_cell(status_item, result)
            self.table.setItem(row, 9, status_item) 

            actions_widget = self._create_action_buttons(row, path)
            self.table.setCellWidget(row, 10, actions_widget)

        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeToContents)
        # Ensure the Actions column stays at a usable width
        self.table.setColumnWidth(12, 60)

    def _handle_hover(self, row, column):
        item = self.table.item(row, 1)
        if not item: return
        file_path = item.data(Qt.UserRole)
        result = self.session.results.get(file_path)
        if not result: return

        pos = QCursor.pos()
        self.magnifier.hide()
        self.chart_magnifier.hide()

        if column == 0: # Sampled Chart Hover
            self.chart_magnifier.result = result
            self.chart_magnifier.move(pos.x() + 20, pos.y() - 200)
            self.chart_magnifier.show()
            self.chart_magnifier.update()
        elif column == 7: # Visual Check Hover
            self.magnifier.result = result
            self.magnifier.move(pos.x() + 20, pos.y() - 150)
            self.magnifier.show()
            self.magnifier.update()

    def leaveEvent(self, event):
        self.magnifier.hide()
        self.chart_magnifier.hide()
        super().leaveEvent(event)

    def _create_input_space_combo(self, row, current_value):
        # 1. Create the shared factory combo
        combo = create_ocio_combo(self.color_engine, current_value)
        
        # Style the combo slightly so it doesn't look cramped
        combo.setMinimumWidth(180)
        combo.setStyleSheet("QComboBox { height: 24px; }")
        
        # Connect the "Dirty" logic
        combo.currentTextChanged.connect(lambda text: self._on_input_space_changed(row, text))
        
        # 2. Wrap it in a container to allow centering
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(combo)
        
        # Remove margins so the layout doesn't add extra space, then center
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignCenter)
        
        return container

    def _on_input_space_changed(self, row, new_text):
        # 1. Get the result object
        item = self.table.item(row, 1) # Filename item
        file_path = item.data(Qt.UserRole)
        result = self.session.results.get(file_path)
        
        if result:
            # 2. Update the result metadata
            result.input_space = new_text
            
            # 3. Mark as Dirty
            result.status = AuditStatus.MANUAL_EDIT
            
            # 4. Refresh the status cell
            status_item = self.table.item(row, 9)
            self._update_status_cell(status_item, result)
            
            print(f"[UI] Input space changed for {result.file_path} -> {new_text}. Status: DIRTY")

    def _create_action_buttons(self, row, file_path):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)

        # 1. Redraw/Edit Bounding Box Button
        edit_btn = QPushButton()
        edit_btn.setIcon(qta.icon('fa5s.vector-square', color='#FFD700'))
        edit_btn.setToolTip("Manually Redraw Bounding Box")
        edit_btn.setFixedSize(30, 30) # Slightly smaller to fit better vertically
        edit_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; } "
                               "QPushButton:hover { background-color: #555; }")
        edit_btn.clicked.connect(lambda: self._on_edit_bb_clicked(file_path))

        # 2. Copy CDL Button
        copy_btn = QPushButton()
        copy_btn.setIcon(qta.icon('fa5s.copy', color='#ADD8E6'))
        edit_btn.setToolTip("Copy ASC-CDL values to clipboard")
        copy_btn.setFixedSize(30, 30)
        copy_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; }"
                               "QPushButton:hover { background-color: #555; }")
        copy_btn.clicked.connect(lambda: self._on_copy_cdl_clicked(file_path))

        # 3. Delete Row Button
        delete_btn = QPushButton()
        delete_btn.setIcon(qta.icon('fa5s.trash-alt', color='#ff6666'))
        delete_btn.setToolTip("Remove from Session")
        delete_btn.setFixedSize(30, 30)
        delete_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; } "
                                 "QPushButton:hover { background-color: #663333; }")
        delete_btn.clicked.connect(lambda: self._on_delete_row_clicked(row, file_path))

        layout.addWidget(edit_btn)
        layout.addWidget(copy_btn)
        layout.addWidget(delete_btn)
        layout.setAlignment(Qt.AlignCenter)
        
        return container

    def _on_edit_bb_clicked(self, file_path):
        """Prepares for the manual corner adjustment window."""
        result = self.session.results.get(file_path)
        if not result: return
        
        print(f"[UI] Opening Manual Redraw for: {file_path}")
        # Next step: Launch the InteractiveCornerWindow(result)

    def _on_copy_cdl_clicked(self, file_path):
        result = self.session.results.get(file_path)
        if not result: return

        cdl_str = (f"Slope: {result.slope[0]:.4f} {result.slope[1]:.4f} {result.slope[2]:.4f} | "
                   f"Offset: {result.offset[0]:.4f} {result.offset[1]:.4f} {result.offset[2]:.4f} | "
                   f"Power: 1.0000 1.0000 1.0000 | "
                   f"Sat: {result.sat:.4f}")

        QApplication.clipboard().setText(cdl_str)

    def _on_delete_row_clicked(self, row, file_path):
        """Removes the item from the session and the UI table."""
        # Remove from session data
        if file_path in self.session.results:
            del self.session.results[file_path]
        
        # Remove from UI
        self.table.removeRow(row)
        print(f"[UI] Removed {file_path} from session.")
        # Note: You may need to refresh_table() if you want row indices to stay perfectly in sync 
        # with the lambda closures, or use a more robust ID system.

    def _setup_review_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background-color: #2b2b2b; border-left: 1px solid #444;")
        layout = QVBoxLayout(sidebar)

        info_group = QGroupBox("INFO")
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(QLabel("Audit Space (Locked):"))
        self.audit_space_display = QComboBox()
        self.audit_space_display.addItems(["ACEScg"])
        self.audit_space_display.setEnabled(False) 
        self.audit_space_display.setStyleSheet("""
            QComboBox { 
                background-color: #333; 
                color: #888; 
                font-style: italic; 
                border: 1px solid #444; 
            }
        """)
        info_layout.addWidget(self.audit_space_display)
        layout.addWidget(info_group)

        # 1. Processing Actions
        proc_group = QGroupBox("ACTIONS")
        proc_layout = QVBoxLayout(proc_group)
        self.reprocess_btn = QPushButton(" REPROCESS DIRTY FILES")
        self.reprocess_btn.setIcon(qta.icon('fa5s.sync', color='#FFD700'))
        self.reprocess_btn.setStyleSheet("background-color: #444; font-weight: bold; height: 40px;")
        self.reprocess_btn.clicked.connect(self._reprocess_dirty)
        proc_layout.addWidget(self.reprocess_btn)
        layout.addWidget(proc_group)

        # 2. Tolerance Adjustment (Live Update)
        tol_group = QGroupBox("AUDIT THRESHOLDS")
        tol_layout = QVBoxLayout(tol_group)
        tol_layout.addWidget(QLabel("DeltaE Tolerance:"))
        self.de_spin = QDoubleSpinBox()
        self.de_spin.setRange(0.1, 10.0)
        self.de_spin.setValue(2.0)
        self.de_spin.valueChanged.connect(self._on_tolerance_changed)
        tol_layout.addWidget(self.de_spin)
        layout.addWidget(tol_group)

        # 3. Export Settings
        exp_group = QGroupBox("EXPORT OPTIONS")
        exp_layout = QVBoxLayout(exp_group)
        self.check_cdl = QCheckBox("Export ASC-CDL (.cdl)")
        self.check_lut = QCheckBox("Export Cube LUT (.cube)")
        self.check_matrix = QCheckBox("Export Matrix (.mtx)")
        self.check_pdf = QCheckBox("Generate PDF Report")
        self.check_csv = QCheckBox("Generate CSV Summary")
        for cb in [self.check_cdl, self.check_lut, self.check_matrix, self.check_pdf, self.check_csv]:
            exp_layout.addWidget(cb)
        layout.addWidget(exp_group)

        self._add_export_directory_ui(layout)

        layout.addStretch()

        # 4. Final Export Button
        self.export_btn = QPushButton("EXPORT VALIDATED ITEMS")
        self.export_btn.setMinimumHeight(60)
        self.export_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        layout.addWidget(self.export_btn)

        self.layout.addWidget(sidebar)

    def _update_status_cell(self, item, result):
        """Applies your custom failure labeling and colors."""
        if not result.is_pass:
            if result.alignment_integrity < 0.05: # Your integrity threshold
                item.setText("FAIL (Geometry)")
                item.setForeground(QColor("#ff6666"))
            else:
                item.setText("FAIL (DeltaE)")
                item.setForeground(QColor("#ffb366"))
        else:
            item.setText("PASS")
            item.setForeground(QColor("#66ff66"))
            
        # Add 'Dirty' override if status is MANUAL_EDIT
        if result.status.name == "MANUAL_EDIT":
            item.setText("DIRTY")
            item.setForeground(QColor("#FFD700"))

    def _on_tolerance_changed(self, val):
        # We will implement the logic to loop through results 
        # and flip is_pass if mean_de > val
        pass
    
    def _add_export_directory_ui(self, parent_layout):
        dir_group = QGroupBox("EXPORT DIRECTORY")
        dir_layout = QVBoxLayout(dir_group)

        # Radio buttons for mode selection
        self.radio_same_source = QRadioButton("Same as Source")
        self.radio_custom_dir = QRadioButton("Custom Directory")
        self.radio_same_source.setChecked(True) # Default

        # Group them so only one can be picked
        self.dir_group_toggle = QButtonGroup(self)
        self.dir_group_toggle.addButton(self.radio_same_source)
        self.dir_group_toggle.addButton(self.radio_custom_dir)

        dir_layout.addWidget(self.radio_same_source)
        dir_layout.addWidget(self.radio_custom_dir)

        # Custom directory selection row (Hidden/Disabled unless 'Custom' is picked)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select folder...")
        self.path_edit.setEnabled(False)
        
        self.browse_btn = QPushButton()
        self.browse_btn.setIcon(qta.icon('fa5s.folder-open'))
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.setEnabled(False)
        self.browse_btn.clicked.connect(self._on_browse_clicked)

        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)
        dir_layout.addLayout(path_row)

        # Connect toggle logic
        self.radio_custom_dir.toggled.connect(lambda checked: self.path_edit.setEnabled(checked))
        self.radio_custom_dir.toggled.connect(lambda checked: self.browse_btn.setEnabled(checked))

        parent_layout.addWidget(dir_group)

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if folder:
            self.path_edit.setText(folder)

    def _reprocess_dirty(self):
        dirty_paths = []
        
        for row in range(self.table.rowCount()):
            # Column 1 still holds our filename/filepath data
            item = self.table.item(row, 1)
            if not item: continue
            
            file_path = item.data(Qt.UserRole)
            result = self.session.results.get(file_path)
            
            # Use the enum for the status check
            if result and result.status == AuditStatus.MANUAL_EDIT:
                dirty_paths.append(file_path)
        
        if not dirty_paths:
            print("[UI] No dirty files to reprocess.")
            return

        print(f"[UI] Reprocessing {len(dirty_paths)} files...")
        
        self.reprocess_btn.setEnabled(False)
        self.reprocess_btn.setText(" PROCESSING...")
        
        # This triggers the core logic to run the audit again
        self.session.run_batch(dirty_paths)
    
if __name__ == "__main__":
    from core.models import AuditResult, AuditStatus
    from core.color_engine import ColorEngine
    import cv2
    import numpy as np
    
    class MockSession:
        def __init__(self):
            self.color_engine = ColorEngine()
            res = AuditResult(file_path="D:/VFX/Shot 01.exr")
            res.camera_make = "ARRI"
            res.camera_model = "ALEXA 35"
            res.width, res.height = 4608, 3164
            res.input_space = "ARRI LogC4"
            res.audit_space = "ACEScg" 
            res.analysis_intent = "MATCH GRADE"
            res.is_pass = True
            res.alignment_integrity = 0.9920
            res.status = AuditStatus.COMPLETE
            res.slope = [1.05, 1.0, 0.95]; res.offset = [0,0,0]; res.sat = 1.0
            
            # MOCK RECTIFIED IMAGE (Placeholder for testing)
            # In real use, sampler.py provides this
            mock_img = np.zeros((800, 1200, 3), dtype=np.uint8)
            cv2.putText(mock_img, "TEST RECTIFIED IMAGE", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            # Draw some mock sample dots
            for y in range(100, 800, 200):
                for x in range(100, 1200, 200):
                    cv2.circle(mock_img, (x, y), 10, (0, 255, 0), -1)
            res.rectified_buffer = mock_img
            
            self.results = {res.file_path: res}

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    mock_session = MockSession()
    window = ReviewWindow(mock_session)
    window.show()
    sys.exit(app.exec())