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

from ui.widgets import create_ocio_combo
from core.color_engine import ColorEngine

class SampledChartDelegate(QStyledItemDelegate):
    """Renders the rectified image with sample dots in the table cell."""
    def sizeHint(self, option, index):
        h = option.rect.height() if option.rect.height() > 0 else 120
        return QSize(int(h * 1.5), h)

    def paint(self, painter, option, index):
        if index.column() == 0:
            result = index.data(Qt.UserRole)
            if result and hasattr(result, 'rectified_buffer') and result.rectified_buffer is not None:
                # Convert numpy uint8 buffer to QImage
                arr = result.rectified_buffer
                h, w, ch = arr.shape
                qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                rect = option.rect.adjusted(4, 4, -4, -4)
                painter.drawPixmap(rect, pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                painter.setPen(QColor("#666666"))
                painter.drawText(option.rect, Qt.AlignCenter, "No Image")
        else:
            super().paint(painter, option, index)

class TrianglePatchDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        if index.column() == 8:
            h = option.rect.height() if option.rect.height() > 0 else 120
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
            
            # Use template-based grid if available, else fallback to 6x4
            cols, rows = 6, 4
            padding = 3
            
            patch_size = (rect.height() - (padding * (rows + 1))) / rows
            grid_w = (patch_size * cols) + (padding * (cols - 1))
            offset_x = rect.x() + (rect.width() - grid_w) / 2
            offset_y = rect.y() + padding

            for r in range(rows):
                for c in range(cols):
                    x = offset_x + (c * (patch_size + padding))
                    y = offset_y + (r * (patch_size + padding))
                    patch_rect = QRectF(x, y, patch_size, patch_size)

                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(120 + (r*20), 100 + (c*10), 150)) 
                    t1 = QPolygonF([patch_rect.topLeft(), patch_rect.topRight(), patch_rect.bottomLeft()])
                    painter.drawPolygon(t1)

                    painter.setBrush(QColor(100 + (r*20), 80 + (c*10), 130))
                    t2 = QPolygonF([patch_rect.bottomRight(), patch_rect.topRight(), patch_rect.bottomLeft()])
                    painter.drawPolygon(t2)
            
            painter.restore()
        else:
            super().paint(painter, option, index)

class ChartMagnifier(QWidget):
    """Large popup for the sampled/rectified image."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setFixedSize(550, 400)
        self.setStyleSheet("background: #1a1a1a; border: 2px solid #FFD700;")
        self.result = None

    def paintEvent(self, event):
        if not self.result or self.result.rectified_buffer is None: return
        painter = QPainter(self)
        
        arr = self.result.rectified_buffer
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        img_rect = self.rect().adjusted(10, 10, -10, -40)
        painter.drawPixmap(img_rect, pixmap.scaled(img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        painter.setPen(QColor("#FFD700"))
        painter.drawText(self.rect().adjusted(0, 0, 0, -10), Qt.AlignHCenter | Qt.AlignBottom, 
                         "Sampled Chart (AI Rectified View)")

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
        self.color_engine = getattr(session_manager, 'color_engine', None) 
        if not self.color_engine and hasattr(session_manager, 'sampler'):
            self.color_engine = session_manager.sampler.color_engine
        self.setWindowTitle("Audit Review & Export")
        self.setMinimumSize(1700, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self._setup_table()
        self._setup_review_sidebar()

        self.table.setMouseTracking(True)
        self.magnifier = GridMagnifier()
        self.chart_magnifier = ChartMagnifier() # Initialize image magnifier
        
        self.table.cellEntered.connect(self._handle_hover)
        self.refresh_table()

    def _setup_table(self):
        self.table = QTableWidget(0, 12)
        headers = [
            "Sampled Chart", "Filename", "Camera Info", "Format", 
            "Resolution", "Input Space", "Audit Space", "Intent", 
            "Visual Check", "Integrity", "Status", "ASC-CDL (SOP)"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Apply both delegates
        self.table.setItemDelegateForColumn(0, SampledChartDelegate(self.table))
        self.table.setItemDelegateForColumn(8, TrianglePatchDelegate(self.table))
        
        header = self.table.horizontalHeader()
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

            self.table.setItem(row, 1, QTableWidgetItem(result.file_path.split("/")[-1]))
            self.table.setItem(row, 2, QTableWidgetItem(f"{result.camera_make} {result.camera_model}"))
            self.table.setItem(row, 3, QTableWidgetItem(result.file_path.split(".")[-1].upper()))
            self.table.setItem(row, 4, QTableWidgetItem(f"{getattr(result, 'width', 0)} x {getattr(result, 'height', 0)}"))
            # --- COLUMN 5: INPUT SPACE DROPDOWN ---
            combo = self._create_input_space_combo(row, result.input_space or "Default")
            self.table.setCellWidget(row, 5, combo)
            self.table.setItem(row, 6, QTableWidgetItem(result.audit_space or "Default"))
            self.table.setItem(row, 7, QTableWidgetItem(result.analysis_intent.upper()))
            self.table.setItem(row, 9, QTableWidgetItem(f"{result.alignment_integrity:.4f}"))

            filename_item = QTableWidgetItem(result.file_path.split("/")[-1])
            filename_item.setData(Qt.UserRole, result.file_path) 
            self.table.setItem(row, 1, filename_item)
            
            status_item = QTableWidgetItem()
            self._update_status_cell(status_item, result)
            self.table.setItem(row, 10, status_item) 

            cdl_text = (f"SLOPE: {result.slope[0]:.4f} {result.slope[1]:.4f} {result.slope[2]:.4f}\n"
                        f"OFFSET: {result.offset[0]:.4f} {result.offset[1]:.4f} {result.offset[2]:.4f}\n"
                        f"SAT: {result.sat:.4f}")
            self.table.setItem(row, 11, QTableWidgetItem(cdl_text))

        for i in range(self.table.columnCount()):
            self.table.setColumnWidth(i, self.table.columnWidth(i) + 30)
        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeToContents)

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
        elif column == 8: # Visual Check Hover
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
            from core.models import AuditStatus
            result.status = AuditStatus.MANUAL_EDIT
            
            # 4. Refresh the status cell (Column 10)
            status_item = self.table.item(row, 10)
            self._update_status_cell(status_item, result)
            
            print(f"[UI] Input space changed for {result.file_path} -> {new_text}. Status: DIRTY")

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
            # Check the status of the result
            item = self.table.item(row, 1)
            file_path = item.data(Qt.UserRole)
            result = self.session.results.get(file_path)
            
            from core.models import AuditStatus
            if result and result.status == AuditStatus.MANUAL_EDIT:
                dirty_paths.append(file_path)
        
        if not dirty_paths:
            print("[UI] No dirty files to reprocess.")
            return

        print(f"[UI] Reprocessing {len(dirty_paths)} files with updated settings...")
        
        # Disable button during processing
        self.reprocess_btn.setEnabled(False)
        self.reprocess_btn.setText(" PROCESSING...")
        
        # Start the batch
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