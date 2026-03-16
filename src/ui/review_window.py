import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTableWidget, QTableWidgetItem, QPushButton, 
                               QGroupBox, QCheckBox, QLabel, QFrame, QHeaderView,
                               QDoubleSpinBox, QComboBox, QFileDialog, QApplication, QStyledItemDelegate)
from PySide6.QtCore import Qt, QSize, QPoint
from PySide6.QtGui import QColor, QPainter, QPolygon
import qtawesome as qta

# --- PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))

class TrianglePatchDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Only draw if we are in the 'Visual Check' column
        if index.column() == 7: 
            # Get the AuditResult for this row
            # (In a real app, we'd pull the actual patch data here)
            rect = option.rect
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw a sample split triangle (Mocking the data for now)
            # Top-Left Triangle (Target/Reference)
            painter.setBrush(QColor(150, 150, 150)) # Replace with ref_rgb
            t1 = QPolygon([rect.topLeft(), rect.topRight(), rect.bottomLeft()])
            painter.drawPolygon(t1)

            # Bottom-Right Triangle (Observed/Source)
            painter.setBrush(QColor(100, 100, 100)) # Replace with src_rgb
            t2 = QPolygon([rect.bottomRight(), rect.topRight(), rect.bottomLeft()])
            painter.drawPolygon(t2)

            painter.restore()
        else:
            super().paint(painter, option, index)

class ReviewWindow(QMainWindow):
    def __init__(self, session_manager):
        super().__init__()
        self.session = session_manager
        self.setWindowTitle("Audit Review & Export")
        self.setMinimumSize(1400, 900)

        # Main Layout (Table Left, Sidebar Right)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self._setup_table()
        self._setup_review_sidebar()
        self.refresh_table()

    def _setup_table(self):
        # Added columns for Camera Info and Visual Verification
        self.table = QTableWidget(0, 10)
        headers = [
            "Rectified", "Filename", "Camera Info", "Format", 
            "Input Space", "Audit Space", "Intent", "Visual Check", "Integrity", "Status"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(80) # Room for triangles
        self.layout.addWidget(self.table, stretch=1)
        self.table.setItemDelegateForColumn(7, TrianglePatchDelegate(self.table))

    def _setup_review_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background-color: #2b2b2b; border-left: 1px solid #444;")
        layout = QVBoxLayout(sidebar)

        # 1. Processing Actions
        proc_group = QGroupBox("ACTIONS")
        proc_layout = QVBoxLayout(proc_group)
        self.reprocess_btn = QPushButton(" REPROCESS DIRTY FILES")
        self.reprocess_btn.setIcon(qta.icon('fa5s.sync', color='#FFD700'))
        self.reprocess_btn.setStyleSheet("background-color: #444; font-weight: bold; height: 40px;")
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

        layout.addStretch()

        # 4. Final Export Button
        self.export_btn = QPushButton("EXPORT VALIDATED ITEMS")
        self.export_btn.setMinimumHeight(60)
        self.export_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        layout.addWidget(self.export_btn)

        self.layout.addWidget(sidebar)

    def refresh_table(self):
        """Populates the table from the SessionManager results."""
        self.table.setRowCount(0)
        for path, result in self.session.results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Filename
            self.table.setItem(row, 1, QTableWidgetItem(result.file_path.split("/")[-1]))
            
            # Camera Info
            cam_info = f"{result.camera_make} {result.camera_model}"
            self.table.setItem(row, 2, QTableWidgetItem(cam_info))

            # Status with Color Logic
            status_item = QTableWidgetItem()
            self._update_status_cell(status_item, result)
            self.table.setItem(row, 9, status_item)

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
    
if __name__ == "__main__":
    from core.models import AuditResult, AuditStatus
    
    # Create a mock session-like object for testing the UI
    class MockSession:
        def __init__(self):
            # Create a fake result to see if the table works
            res = AuditResult(file_path="C:/Test/Plate_v01.exr")
            res.camera_make = "ARRI"
            res.camera_model = "Alexa 35"
            res.is_pass = True
            res.status = AuditStatus.COMPLETE
            self.results = {res.file_path: res}

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Pass the mock session here
    mock_session = MockSession()
    window = ReviewWindow(mock_session)
    
    window.show()
    sys.exit(app.exec())