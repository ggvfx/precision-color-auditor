import sys
import os
from pathlib import Path
import numpy as np
import qtawesome as qta
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTableWidget, QHeaderView, QComboBox, 
                             QLabel, QFrame, QFileDialog, QTableWidgetItem, 
                             QApplication, QCheckBox, QLineEdit, QRadioButton, 
                             QGroupBox, QDoubleSpinBox, QButtonGroup, QToolButton)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon

# --- PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
project_root = src_dir.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path: sys.path.insert(0, str(src_dir))

try:
    from core.color_engine import ColorEngine
    from core.ingest import ImageIngestor
    from core.templates import CHART_LIBRARY
    from core.config import settings
    from exporters.utils import get_system_metadata 
except ImportError as e:
    print(f"Critical Import Error: {e}")
    sys.exit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = ColorEngine()
        self.setWindowTitle("Precision Color Auditor - Setup")
        self.setMinimumSize(1300, 850)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self._setup_sidebar()
        self._setup_queue_area()
        self._setup_status_bar()

    def _setup_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setObjectName("Sidebar")
        sidebar.setStyleSheet("QFrame#Sidebar { background-color: #2b2b2b; border-right: 1px solid #444; }")
        layout = QVBoxLayout(sidebar)

        # 1. INGEST CONTROLS (At the top)
        ingest_group = QGroupBox("INGEST CONTROLS")
        ing_layout = QVBoxLayout(ingest_group)
        
        btn_hbox = QHBoxLayout()
        self.add_file_btn = QPushButton("+ File")
        self.add_folder_btn = QPushButton("+ Folder")
        self.add_file_btn.clicked.connect(lambda: self._handle_ingest(is_folder=False))
        self.add_folder_btn.clicked.connect(lambda: self._handle_ingest(is_folder=True))
        btn_hbox.addWidget(self.add_file_btn)
        btn_hbox.addWidget(self.add_folder_btn)
        ing_layout.addLayout(btn_hbox)

        self.recurse_check = QCheckBox("Recurse Subfolders")
        self.recurse_check.setChecked(True)
        ing_layout.addWidget(self.recurse_check)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by name...")
        ing_layout.addWidget(self.filter_input)
        
        self.clear_table_btn = QPushButton(" Clear Selection")
        self.clear_table_btn.setIcon(qta.icon('fa5s.trash-alt', color='#ff6666'))
        self.clear_table_btn.setStyleSheet("color: #ff6666; text-align: left; padding: 5px;")
        self.clear_table_btn.clicked.connect(lambda: self.table.setRowCount(0))
        ing_layout.addWidget(self.clear_table_btn)
        layout.addWidget(ingest_group)

        # 2. ANALYSIS INTENT
        intent_group = QGroupBox("ANALYSIS INTENT")
        intent_layout = QHBoxLayout(intent_group)
        self.intent_group = QButtonGroup(self)
        self.radio_neutral = QRadioButton("Neutralize")
        self.radio_match = QRadioButton("Match Grade")
        
        self.radio_neutral.setChecked(settings.analysis_intent == "neutralize")
        self.radio_match.setChecked(settings.analysis_intent == "match_grade")
            
        self.intent_group.addButton(self.radio_neutral)
        self.intent_group.addButton(self.radio_match)
        intent_layout.addWidget(self.radio_neutral)
        intent_layout.addWidget(self.radio_match)
        layout.addWidget(intent_group)

        # 3. OCIO COLOR PIPELINE (Combined)
        pipeline_group = QGroupBox("OCIO COLOR PIPELINE")
        pipe_layout = QVBoxLayout(pipeline_group)
        
        self.ocio_path_label = QLabel(str(settings.current_ocio_path.name))
        self.ocio_path_label.setStyleSheet("font-size: 10px; color: #888; margin-bottom: 2px;")
        pipe_layout.addWidget(self.ocio_path_label)
        
        self.ocio_btn = QPushButton("Change OCIO Config...")
        self.ocio_btn.clicked.connect(self._browse_ocio)
        pipe_layout.addWidget(self.ocio_btn)

        pipe_layout.addWidget(QLabel("Input Space (Source):"))
        self.src_space_combo = QComboBox()
        pipe_layout.addWidget(self.src_space_combo)

        pipe_layout.addWidget(QLabel("Audit Space (Locked):"))
        self.audit_space_display = QLabel("ACEScg")
        self.audit_space_display.setStyleSheet("background-color: #1a1a1a; color: #00ff00; padding: 5px; border: 1px solid #444; font-family: monospace;")
        pipe_layout.addWidget(self.audit_space_display)

        pipe_layout.addWidget(QLabel("Display Space (UI):"))
        self.display_space_combo = QComboBox()
        pipe_layout.addWidget(self.display_space_combo)
        layout.addWidget(pipeline_group)
        
        self._populate_spaces() # Defaults applied here

        self.adv_header = QPushButton(" Advanced Settings")
        self.adv_header.setIcon(qta.icon('fa5s.caret-right', color='#888'))
        self.adv_header.setCheckable(True)
        self.adv_header.setStyleSheet("QPushButton { text-align: left; border: none; font-weight: bold; background: #333; padding: 5px; }")
        layout.addWidget(self.adv_header)

        self.adv_container = QFrame()
        self.adv_container.setVisible(False)
        adv_layout = QVBoxLayout(self.adv_container)
        
        # Move existing DeltaE and Output logic into this layout
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("DeltaE Tol:"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setValue(settings.tolerance_threshold)
        tol_layout.addWidget(self.tol_spin)
        adv_layout.addLayout(tol_layout)

        self.out_btn = QPushButton(f"Out: {settings.output_dir.name}...")
        self.out_btn.clicked.connect(self._browse_output)
        adv_layout.addWidget(self.out_btn)
        
        layout.addWidget(self.adv_container)

        # Toggle Connection
        def toggle_adv(checked):
            self.adv_container.setVisible(checked)
            icon_name = 'fa5s.caret-down' if checked else 'fa5s.caret-right'
            self.adv_header.setIcon(qta.icon(icon_name, color='#888'))
        
        self.adv_header.toggled.connect(toggle_adv)

        layout.addStretch()
        
        self.process_btn = QPushButton("PROCEED TO AUDIT")
        self.process_btn.setMinimumHeight(60)
        self.process_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        self.process_btn.setEnabled(False) 
        layout.addWidget(self.process_btn)

        self.layout.addWidget(sidebar)

    def _populate_spaces(self):
        src_list, audit_list = self.engine.get_ui_lists()
        
        # Block signals to prevent triggering change events while populating
        self.src_space_combo.blockSignals(True)
        self.src_space_combo.clear()
        self.src_space_combo.addItems(src_list)
        
        #self.audit_space_combo.clear()
        #self.audit_space_combo.addItems(audit_list)
        
        self.display_space_combo.clear()
        self.display_space_combo.addItems(src_list)

        # APPLY CONFIG DEFAULTS
        self.src_space_combo.setCurrentText(settings.default_input_space)
        #self.audit_space_combo.setCurrentText(settings.default_audit_space)
        self.display_space_combo.setCurrentText(settings.default_display_space)
        
        self.src_space_combo.blockSignals(False)

    def _setup_queue_area(self):
        queue_container = QVBoxLayout()
        # Col 4 renamed to "Signal Profile"
        # Col 6 added for Remove Button
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Thumb", "Filename", "Format", "Resolution", "Signal Profile", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 65)
        self.table.setColumnWidth(6, 40)
        self.table.setIconSize(QSize(60, 45))
        queue_container.addWidget(self.table)
        self.layout.addLayout(queue_container, stretch=1)

    def _handle_ingest(self, is_folder=False):
        valid_exts = ['.exr', '.jpg', '.png', '.cr2', '.arw', '.dng', '.dpx']
        files = []
        filter_str = self.filter_input.text().lower()

        if is_folder:
            folder = QFileDialog.getExistingDirectory(self, "Select Ingest Folder")
            if folder:
                path_obj = Path(folder)
                pattern = "**/*" if self.recurse_check.isChecked() else "*"
                files = [str(f) for f in path_obj.glob(pattern) if f.suffix.lower() in valid_exts and filter_str in f.name.lower()]
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.exr *.jpg *.png *.cr2 *.arw)")
            if path: files = [path]

        for f_path in files:
            self._add_to_table(f_path)

    def _add_to_table(self, path):
        try:
            pixels, meta = ImageIngestor.load_image(path)
            row = self.table.rowCount()
            self.table.insertRow(row)

            # 1. Thumbnail
            thumb_data = (np.clip(pixels[::pixels.shape[0]//45, ::pixels.shape[1]//60], 0, 1) * 255).astype('uint8')
            h, w, c = thumb_data.shape
            qimg = QImage(thumb_data.data, w, h, w*c, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            thumb_label = QLabel()
            thumb_label.setPixmap(pix.scaled(60, 45, Qt.KeepAspectRatio))
            thumb_label.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(row, 0, thumb_label)

            # 2. Text Metadata
            self.table.setItem(row, 1, QTableWidgetItem(os.path.basename(path)))
            self.table.setItem(row, 2, QTableWidgetItem(meta['file_format']))
            self.table.setItem(row, 3, QTableWidgetItem(f"{meta['width']}x{meta['height']}"))
            self.table.setItem(row, 5, QTableWidgetItem("Ready"))

            # 3. Signal Profile Dropdown with Strict Waterfall
            profile_combo = QComboBox()
            src_list, _ = self.engine.get_ui_lists()
            profile_combo.addItems(src_list)
            
            raw_hint = meta.get('colorspace_hint', "").strip().lower()
            best_match = None

            # Determine our primary search target based on metadata hint
            target = None
            if "srgb" in raw_hint:
                target = "sRGB - Texture"
            elif "linear" in raw_hint or "acescg" in raw_hint:
                target = "ACEScg"

            if target:
                target_l = target.lower()
                # 1. Exact Match
                exact = next((s for s in src_list if s.lower() == target_l), None)
                # 2. Starts With
                starts = next((s for s in src_list if s.lower().startswith(target_l)), None)
                # 3. Contains
                contains = next((s for s in src_list if target_l in s.lower()), None)

                best_match = exact or starts or contains

            if best_match:
                profile_combo.setCurrentText(best_match)
            else:
                # Fallback if target logic fails or no matches found
                profile_combo.insertItem(0, "Unknown")
                profile_combo.setCurrentIndex(0)
                profile_combo.setStyleSheet("QComboBox { color: #ff6666; border: 1px solid #ff6666; }")
            
            self.table.setCellWidget(row, 4, profile_combo)

            # 4. Corrected Delete Button
            del_btn = QToolButton()
            del_btn.setIcon(qta.icon('fa5s.times-circle', color='#888'))
            del_btn.setStyleSheet("border: none;")
            
            # Using a lambda with the button reference is safer here
            del_btn.clicked.connect(lambda: self._remove_row_logic(del_btn))
            self.table.setCellWidget(row, 6, del_btn)
            
            self.process_btn.setEnabled(True)

        except Exception as e:
            print(f"Ingest Error: {e}")

    def _remove_row_logic(self, btn):
        # Reliable way to find the row even after sorting or multiple deletions
        index = self.table.indexAt(btn.pos())
        if index.isValid():
            self.table.removeRow(index.row())
            if self.table.rowCount() == 0:
                self.process_btn.setEnabled(False)

    def _browse_ocio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select OCIO Config", "", "OCIO Config (*.ocio)")
        if path:
            self.engine.initialize_config(path)
            settings.update_ocio_config(path)
            self.ocio_path_label.setText(Path(path).name)
            self._populate_spaces()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(settings.output_dir))
        if path:
            settings.output_dir = Path(path)
            self.out_btn.setText(f"Out: {settings.output_dir.name}...")

    def _setup_status_bar(self):
        stat = self.statusBar()
        sys_info = get_system_metadata()
        stat.showMessage(f"Host: {sys_info['hostname']} | User: {sys_info['user']} | OS: {sys_info['os']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())