import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTableWidget, QTableWidgetItem, QPushButton, 
                               QGroupBox, QCheckBox, QLabel, QFrame, QHeaderView,
                               QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
                               QFileDialog, QApplication, QStyledItemDelegate, QLineEdit)
from PySide6.QtCore import Qt, QSize, QPoint, QRectF
from PySide6.QtGui import QColor, QPainter, QPolygon, QPolygonF, QCursor
import qtawesome as qta

# --- PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))

class TrianglePatchDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        if index.column() == 8:
            # Get the current row height
            h = option.rect.height() if option.rect.height() > 0 else 120
            
            # Macbeth is 6 cols, 4 rows. 
            # To keep patches square: 
            # patch_height = (h - vertical_padding) / 4
            # total_width = (patch_height * 6) + horizontal_padding
            padding = 3
            patch_h = (h - (padding * 5)) / 4
            total_w = (patch_h * 6) + (padding * 7)
            
            return QSize(total_w, h)
        return super().sizeHint(option, index)

    def paint(self, painter, option, index):
        if index.column() == 8:
            rect = option.rect
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            
            cols, rows = 6, 4
            padding = 3
            
            # Calculate patch size based on the height of the cell provided by the table
            patch_size = (rect.height() - (padding * (rows + 1))) / rows
            
            # Center the grid horizontally in case the column is wider than required
            grid_w = (patch_size * cols) + (padding * (cols - 1))
            offset_x = rect.x() + (rect.width() - grid_w) / 2
            offset_y = rect.y() + padding

            for r in range(rows):
                for c in range(cols):
                    x = offset_x + (c * (patch_size + padding))
                    y = offset_y + (r * (patch_size + padding))
                    patch_rect = QRectF(x, y, patch_size, patch_size)

                    # Top-Left: Target
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(120 + (r*20), 100 + (c*10), 150)) 
                    t1 = QPolygonF([patch_rect.topLeft(), patch_rect.topRight(), patch_rect.bottomLeft()])
                    painter.drawPolygon(t1)

                    # Bottom-Right: Observed
                    painter.setBrush(QColor(100 + (r*20), 80 + (c*10), 130))
                    t2 = QPolygonF([patch_rect.bottomRight(), patch_rect.topRight(), patch_rect.bottomLeft()])
                    painter.drawPolygon(t2)
            
            painter.restore()
        else:
            super().paint(painter, option, index)

class GridMagnifier(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        # 1. Tighter overall window size
        self.setFixedSize(450, 320) 
        self.setStyleSheet("background: #1a1a1a; border: 2px solid #555;")
        self.result = None

    def paintEvent(self, event):
        if not self.result: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 2. Adjusted Grid Area: Leaves 40px at the bottom for the text
        # (top, left, bottom, right)
        grid_area = self.rect().adjusted(10, 10, -10, -40)
        cols, rows = 6, 4
        padding = 5
        
        # Calculate size based on the restricted grid_area
        patch_size = min((grid_area.width() - (padding*7))/6, (grid_area.height() - (padding*5))/4)
        
        # Center horizontally in the widget
        grid_w = (patch_size * cols) + (padding * (cols - 1))
        offset_x = (self.width() - grid_w) / 2

        for r in range(rows):
            for c in range(cols):
                x = offset_x + (c * (patch_size + padding))
                y = grid_area.y() + (r * (patch_size + padding))
                patch_rect = QRectF(x, y, patch_size, patch_size)
                
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(120 + (r*20), 100 + (c*10), 150))
                t1 = QPolygonF([patch_rect.topLeft(), patch_rect.topRight(), patch_rect.bottomLeft()])
                painter.drawPolygon(t1)
                
                painter.setBrush(QColor(100 + (r*20), 80 + (c*10), 130))
                t2 = QPolygonF([patch_rect.bottomRight(), patch_rect.topRight(), patch_rect.bottomLeft()])
                painter.drawPolygon(t2)

        # 3. Draw Legend (Now strictly within the bottom 40px)
        legend_rect = QRectF(0, self.height() - 40, self.width(), 30)
        painter.setPen(QColor("#CCCCCC"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        intent = getattr(self.result, 'analysis_intent', "MATCH GRADE").upper()
        if "MATCH" in intent:
            legend_text = "Left: Match Grade (Corrected Target)  |  Right: Input Plate (Observed)"
        else:
            legend_text = "Left: Neutralized Plate (Corrected)  |  Right: Target Values (Reference)"

        painter.drawText(legend_rect, Qt.AlignCenter, legend_text)

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

        # Mouse tracking for magnification on hover
        self.table.setMouseTracking(True)
        self.magnifier = GridMagnifier()
        # Connect the cellEntered signal
        self.table.cellEntered.connect(self._handle_hover)

        self.refresh_table()

    def _setup_table(self):
        self.table = QTableWidget(0, 12)
        headers = [
            "Rectified", "Filename", "Camera Info", "Format", 
            "Resolution", "Input Space", "Audit Space", "Intent", 
            "Visual Check", "Integrity", "Status", "ASC-CDL (SOP)"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setItemDelegateForColumn(8, TrianglePatchDelegate(self.table))
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Apply the delegate and set a taller row for the grid
        self.table.verticalHeader().setDefaultSectionSize(120)
        self.layout.addWidget(self.table, stretch=1)

    def refresh_table(self):
        self.table.setRowCount(0)
        for path, result in self.session.results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Metadata (0-4)
            self.table.setItem(row, 1, QTableWidgetItem(result.file_path.split("/")[-1]))
            self.table.setItem(row, 2, QTableWidgetItem(f"{result.camera_make} {result.camera_model}"))
            self.table.setItem(row, 3, QTableWidgetItem(result.file_path.split(".")[-1].upper()))
            self.table.setItem(row, 4, QTableWidgetItem(f"{getattr(result, 'width', 0)} x {getattr(result, 'height', 0)}"))

            # Spaces (5-7)
            self.table.setItem(row, 5, QTableWidgetItem(result.input_space or "Default"))
            self.table.setItem(row, 6, QTableWidgetItem(result.audit_space or "Default"))
            self.table.setItem(row, 7, QTableWidgetItem(result.analysis_intent.upper()))

            # Quality & Status (9-10) - FIXED INDEXING
            self.table.setItem(row, 9, QTableWidgetItem(f"{result.alignment_integrity:.4f}"))

            # magnification Lookup
            filename_item = QTableWidgetItem(result.file_path.split("/")[-1])
            filename_item.setData(Qt.UserRole, result.file_path) # Store the full path for lookup
            self.table.setItem(row, 1, filename_item)
            
            status_item = QTableWidgetItem()
            self._update_status_cell(status_item, result)
            self.table.setItem(row, 10, status_item) 

            # CDL (11)
            cdl_text = (f"SLOPE: {result.slope[0]:.4f} {result.slope[1]:.4f} {result.slope[2]:.4f}\n"
                        f"OFFSET: {result.offset[0]:.4f} {result.offset[1]:.4f} {result.offset[2]:.4f}\n"
                        f"SAT: {result.sat:.4f}")
            self.table.setItem(row, 11, QTableWidgetItem(cdl_text))

        # After data is in, force a small extra buffer to widths to prevent cramping
        for i in range(self.table.columnCount()):
            self.table.setColumnWidth(i, self.table.columnWidth(i) + 30)

        # Force the header to recalculate based on the delegate sizeHints
        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeToContents)

    def _handle_hover(self, row, column):
        if column == 8:
            # Get the path we stored in UserRole
            item = self.table.item(row, 1)
            if not item: return
            
            file_path = item.data(Qt.UserRole)
            result = self.session.results.get(file_path)
            
            # Position the magnifier next to the cursor
            pos = QCursor.pos()
            self.magnifier.move(pos.x() + 20, pos.y() - 150)
            self.magnifier.result = result
            self.magnifier.show()
            self.magnifier.update()
        else:
            if hasattr(self, 'magnifier'):
                self.magnifier.hide()

    # Also hide it when the mouse leaves the table entirely
    def leaveEvent(self, event):
        self.magnifier.hide()
        super().leaveEvent(event)

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
    
if __name__ == "__main__":
    from core.models import AuditResult, AuditStatus
    
    class MockSession:
        def __init__(self):
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
            
            # Using actual values from your report 
            res.slope = [1.0500, 1.0000, 0.9500] 
            res.offset = [0.0000, 0.0000, 0.0000]
            res.sat = 1.0000
            
            self.results = {res.file_path: res}

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    mock_session = MockSession()
    window = ReviewWindow(mock_session)
    window.show()
    sys.exit(app.exec())