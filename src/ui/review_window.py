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
    GridMagnifier,
    ActionButtons,
    ReviewSidebar
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

        # 1. Create the base widgets (No inter-dependencies yet)
        # 1. Create the widgets in the original order for the layout
        self._setup_table()           # Table first (Left side)
        self._setup_review_sidebar()  # Sidebar second (Right side)

        initial_tol = getattr(self.session, 'delta_e_tolerance', 2.0)
        self.de_spin.setValue(initial_tol)

        # 2. NOW assign the Lens-aware delegate (Surgical Fix)
        # We do this here because both 'self.table' and 'self.view_transform_combo' now exist
        self.chart_delegate = SampledChartDelegate(self.table, self.color_engine, self.view_transform_combo)
        self.table.setItemDelegateForColumn(6, self.chart_delegate)
        self.patch_delegate = TrianglePatchDelegate(self.table, self.color_engine, self.view_transform_combo)
        self.table.setItemDelegateForColumn(7, self.patch_delegate)
        
        # 3. Connect the "Lens" signal
        self.view_transform_combo.currentTextChanged.connect(self.table.viewport().update)
        self.magnifier = GridMagnifier(color_engine=self.color_engine)
        self.chart_magnifier = ChartMagnifier(color_engine=self.color_engine) 
        
        # 2. Finalize the Bridge (Now that both table and sidebar exist)
        self._finalize_ui_connections()

        self.table.setMouseTracking(True)
        
        self.table.cellEntered.connect(self._handle_hover)
        self.refresh_table()

    def _setup_table(self):
        self.table = QTableWidget(0, 11)
        headers = [
            "Filename", "Camera Info", "Format", 
            "Resolution", "Input Space", "Intent", "Sampled Chart", 
            "Visual Check", "Integrity", "Status", "Actions"
        ]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Apply both delegates
        #self.table.setItemDelegateForColumn(6, SampledChartDelegate(self.table, self.color_engine, self.view_transform_combo))
        self.table.setItemDelegateForColumn(7, TrianglePatchDelegate(self.table))
        
        header = self.table.horizontalHeader()
        self.table.setColumnWidth(10, 50)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(120)
        self.layout.addWidget(self.table, stretch=1)

    def refresh_table(self):
        self.table.setRowCount(0)
        for path, result in self.session.results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
 
            # Col 0: Filename
            filename_item = QTableWidgetItem(result.file_path.split("/")[-1])
            filename_item.setData(Qt.UserRole, result.file_path) 
            self.table.setItem(row, 0, filename_item)
            # Col 1: Camera Info
            self.table.setItem(row, 1, QTableWidgetItem(f"{result.camera_make} {result.camera_model}"))
            # Col 2: Format
            self.table.setItem(row, 2, QTableWidgetItem(result.file_path.split(".")[-1].upper()))
            # Col 3: Resolution
            self.table.setItem(row, 3, QTableWidgetItem(f"{getattr(result, 'width', 0)} x {getattr(result, 'height', 0)}"))
            # Col 4: Input Space Dropdown
            combo = self._create_input_space_combo(row, result.input_space or "Default")
            self.table.setCellWidget(row, 4, combo)
            # Col 5: Intent
            self.table.setItem(row, 5, QTableWidgetItem(result.analysis_intent.upper()))
            # Col 6: Sampled Chart
            rect_item = QTableWidgetItem()
            rect_item.setData(Qt.UserRole, result)
            self.table.setItem(row, 6, rect_item)
            # Col 7: Visual Check
            self.table.setItem(row, 7, QTableWidgetItem())
            # Col 8: Integrity
            self.table.setItem(row, 8, QTableWidgetItem(f"{result.alignment_integrity:.4f}"))
            # Col 9: Status
            status_item = QTableWidgetItem()
            self._update_status_cell(status_item, result)
            self.table.setItem(row, 9, status_item) 
            # Col 10: Actions
            actions_widget = self._create_action_buttons(row, path)
            self.table.setCellWidget(row, 10, actions_widget)

        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeToContents)
        self.table.setColumnWidth(6, 180)
        # Ensure the Actions column stays at a usable width
        self.table.setColumnWidth(10, 60)

    def _handle_hover(self, row, column):
        item = self.table.item(row, 0)
        if not item: return
        file_path = item.data(Qt.UserRole)
        result = self.session.results.get(file_path)
        if not result: return

        pos = QCursor.pos()
        self.magnifier.hide()
        self.chart_magnifier.hide()

        current_view = self.view_transform_combo.currentText()

        if column == 6: # Sampled Chart Hover
            self.chart_magnifier.result = result
            self.chart_magnifier.view_space = current_view
            self.chart_magnifier.move(pos.x() + 20, pos.y() - 200)
            self.chart_magnifier.show()
            self.chart_magnifier.update()
        elif column == 7: # Visual Check Hover
            self.magnifier.result = result
            self.magnifier.view_space = current_view
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
        item = self.table.item(row, 0) # Filename item
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
        actions = ActionButtons()
        
        # Connect the internal widget signals to the window methods
        actions.edit_clicked.connect(lambda: self._on_edit_bb_clicked(file_path))
        actions.copy_clicked.connect(lambda: self._on_copy_cdl_clicked(file_path))
        actions.delete_clicked.connect(lambda: self._on_delete_row_clicked(row, file_path))
        
        return actions

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
        # 1. Instantiate the new widget
        self.sidebar = ReviewSidebar(self.color_engine, self.session)
        
        # 2. Re-point internal references so your existing methods don't break
        # (This is the secret to not having to rewrite all your methods)
        self.view_transform_combo = self.sidebar.view_transform_combo
        self.de_spin = self.sidebar.de_spin
        self.reprocess_btn = self.sidebar.reprocess_btn
        
        # 3. Connect signals to your existing window methods
        self.sidebar.tolerance_changed.connect(self._on_tolerance_changed)
        self.sidebar.reprocess_requested.connect(self._reprocess_dirty)
        self.sidebar.view_changed.connect(self.table.viewport().update)
        
        # 4. Add the Export Directory UI to the sidebar's dedicated layout
        self._add_export_directory_ui(self.sidebar.export_dir_layout)
        
        # 5. Add to the main window layout
        self.layout.addWidget(self.sidebar)

    def _finalize_ui_connections(self):
        # 1. Now safely connect the Sidebar to the Table Viewport
        self.view_transform_combo.currentTextChanged.connect(self.table.viewport().update)

        # 2. Now safely assign the Lens-aware delegates
        self.chart_delegate = SampledChartDelegate(self.table, self.color_engine, self.view_transform_combo)
        self.table.setItemDelegateForColumn(6, self.chart_delegate)
        
        # (Task 4 will go here next)
        # self.patch_delegate = TrianglePatchDelegate(...)

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

    def _on_tolerance_changed(self, new_tolerance):
        """Live-updates the PASS/FAIL status without reprocessing pixels."""
        for row in range(self.table.rowCount()):
            # 1. Recover the result object from the hidden data in Column 0
            item = self.table.item(row, 0)
            if not item: continue
            
            file_path = item.data(Qt.UserRole)
            result = self.session.results.get(file_path)
            
            if result:
                # 2. Re-calculate the PASS/FAIL flag based on the new threshold
                # Logic: Result is a pass only if the mean error is below the spinbox value
                result.is_pass = result.delta_e_mean <= new_tolerance
                
                # 3. Trigger a visual refresh of the Status cell (Column 9)
                status_item = self.table.item(row, 9)
                if status_item:
                    self._update_status_cell(status_item, result)
        
        print(f"[UI] Batch re-validated against DeltaE: {new_tolerance}")
    
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
    from core.models import AuditResult, AuditStatus, ColorPatch
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
            res.display_space = "sRGB - Texture" # Set this to fix the inheritance bug
            res.analysis_intent = "MATCH GRADE"
            res.is_pass = True
            res.alignment_integrity = 0.9920
            res.delta_e_mean = 1.2
            res.status = AuditStatus.COMPLETE
            res.slope = [1.05, 1.0, 0.95]; res.offset = [0,0,0]; res.sat = 1.0
            
            mock_img = np.zeros((800, 1200, 3), dtype=np.uint8)
            cv2.putText(mock_img, "TEST RECTIFIED IMAGE", (300, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            res.patch_centers = []
            res.rectified_buffer = mock_img 

            for y in range(100, 800, 200):
                for x in range(100, 1200, 200):
                    res.patch_centers.append((x, y))

            # Standard Macbeth 24 sRGB approximations
            macbeth_colors = [
                [0.45, 0.32, 0.24], [0.76, 0.58, 0.50], [0.36, 0.48, 0.61], [0.35, 0.43, 0.25],
                [0.50, 0.50, 0.73], [0.38, 0.74, 0.66], [0.85, 0.48, 0.18], [0.28, 0.36, 0.65],
                [0.75, 0.35, 0.39], [0.36, 0.23, 0.42], [0.62, 0.74, 0.25], [0.90, 0.63, 0.16],
                [0.05, 0.19, 0.43], [0.26, 0.58, 0.30], [0.69, 0.22, 0.26], [0.94, 0.86, 0.12],
                [0.72, 0.33, 0.57], [0.00, 0.52, 0.65], [0.95, 0.95, 0.95], [0.78, 0.78, 0.78],
                [0.62, 0.62, 0.62], [0.47, 0.47, 0.47], [0.33, 0.33, 0.33], [0.13, 0.13, 0.13]
            ]

            res.patches = []
            res.patch_centers = []
            for i in range(24):
                v_ref = np.array(macbeth_colors[i], dtype=np.float32)
                v_src = v_ref.copy()
                if i < 18: v_src *= 0.92 # Visual offset for testing

                row_idx = i // 6
                col_idx = i % 6
                center_x = 100 + (col_idx * 200)
                center_y = 100 + (row_idx * 200)
                
                res.patch_centers.append((center_x, center_y))
                
                res.patches.append(ColorPatch(
                    name=f"Macbeth_{i}",
                    observed_rgb=v_src,
                    target_rgb=v_ref,
                    local_center=(0, 0),
                    index=i,
                    visual_src_rgb=v_src,
                    visual_ref_rgb=v_ref
                ))
            self.results = {res.file_path: res}



    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    mock_session = MockSession()
    window = ReviewWindow(mock_session)
    window.show()
    sys.exit(app.exec())