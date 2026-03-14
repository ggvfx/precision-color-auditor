"""
Primary Calibration Interface & Orchestrator.

A monolithic window handling the viewer, control panels, and signal logic.
Manages the direct interaction between PySide6 widgets and the core color engine.
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTableWidget, QHeaderView, QComboBox, 
                             QLabel, QFrame, QFileDialog, QTableWidgetItem, QApplication)
from PySide6.QtCore import Qt

# --- ROBUST PATH RESOLUTION ---
# 1. Get the directory where this script lives (src/ui)
current_dir = Path(__file__).resolve().parent
# 2. Get the 'src' directory
src_dir = current_dir.parent
# 3. Get the project root directory
project_root = src_dir.parent

# Add both to sys.path to support 'import core' AND 'import src.core'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# --- NOW IMPORTS WILL WORK ---
try:
    from core.color_engine import ColorEngine
    from core.ingest import ImageIngestor
    from core.templates import CHART_LIBRARY
    # Note: Using the style consistent with your other core modules
    from exporters.utils import get_system_metadata 
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print(f"Path Check: {sys.path[:2]}")
    sys.exit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = ColorEngine()
        self.setWindowTitle("Precision Color Auditor - Setup")
        self.setMinimumSize(1100, 700)
        
        # Central Hub
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)

        self._setup_sidebar()
        self._setup_queue_area()
        self._setup_status_bar()

    def _setup_sidebar(self):
        """Zone A: Global Configuration Sidebar."""
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setObjectName("Sidebar")
        layout = QVBoxLayout(sidebar)

        layout.addWidget(QLabel("<b>COLOR PIPELINE</b>"))
        
        # OCIO Config Selection
        self.ocio_btn = QPushButton("Load OCIO Config...")
        self.ocio_btn.clicked.connect(self._browse_ocio)
        layout.addWidget(self.ocio_btn)

        # Audit Space Selection (Filtered Linear Spaces)
        layout.addWidget(QLabel("Audit Space (Linear):"))
        self.audit_space_combo = QComboBox()
        _, audit_list = self.engine.get_ui_lists()
        self.audit_space_combo.addItems(audit_list)
        layout.addWidget(self.audit_space_combo)

        layout.addSpacing(20)
        layout.addWidget(QLabel("<b>CALIBRATION TARGET</b>"))
        
        # Template Selection
        self.template_combo = QComboBox()
        from core.templates import CHART_LIBRARY
        self.template_combo.addItems([CHART_LIBRARY[k].label for k in CHART_LIBRARY])
        layout.addWidget(self.template_combo)

        layout.addStretch()
        
        # Process Button (Navigation to Window 2)
        self.process_btn = QPushButton("PROCEED TO AUDIT")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setEnabled(False) 
        layout.addWidget(self.process_btn)

        self.layout.addWidget(sidebar)

    def _setup_queue_area(self):
        """Zone B: File Ingest Queue."""
        queue_container = QVBoxLayout()
        
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Filename", "Format", "Resolution", "Color Hint", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        
        queue_container.addWidget(self.table)
        self.layout.addLayout(queue_container, stretch=1)

    def _setup_status_bar(self):
        """Zone C: Environment Metadata."""
        stat = self.statusBar()
        from src.exporters.utils import get_system_metadata
        sys_info = get_system_metadata()
        
        info_str = f"User: {sys_info['user']}  |  Host: {sys_info['hostname']}  |  Platform: {sys_info['os']}"
        stat.showMessage(info_str)

    def _browse_ocio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select OCIO Config", "", "OCIO Config (*.ocio)")
        if path:
            self.engine.initialize_config(path)
            _, audit_list = self.engine.get_ui_lists()
            self.audit_space_combo.clear()
            self.audit_space_combo.addItems(audit_list)

    def _test_add_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.exr *.jpg *.png *.cr2 *.arw)")
        if path:
            try:
                # Use your Ingestor to peek at the file
                _, meta = ImageIngestor.load_image(path)
                
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
                self.table.setItem(row, 1, QTableWidgetItem(meta['file_format']))
                self.table.setItem(row, 2, QTableWidgetItem(f"{meta['width']}x{meta['height']}"))
                self.table.setItem(row, 3, QTableWidgetItem(meta['colorspace_hint']))
                self.table.setItem(row, 4, QTableWidgetItem("Ready"))
                
                self.process_btn.setEnabled(True)
            except Exception as e:
                print(f"Ingest Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Dark-friendly theme
    
    # Simple Dark Mode Palette
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

