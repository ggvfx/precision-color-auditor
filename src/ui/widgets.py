from PySide6.QtWidgets import (QComboBox, QWidget, QStyledItemDelegate, 
                               QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                               QFrame, QGroupBox, QCheckBox, QDoubleSpinBox, 
                               QRadioButton, QButtonGroup, QLineEdit)
from PySide6.QtCore import Qt, QSize, QRectF, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPolygonF, QImage, QPixmap
import numpy as np
import qtawesome as qta

def create_ocio_combo(color_engine, current_value=None, is_fallback=False):
    """Standardized factory for OCIO input space dropdowns."""
    combo = QComboBox()
    src_list, _ = color_engine.get_ui_lists()
    combo.addItems(src_list)
    if current_value:
        if current_value not in src_list:
            combo.insertItem(0, current_value)
        combo.setCurrentText(current_value)

    base_style = "QComboBox { background-color: #333; padding: 2px; "
    if is_fallback:
        base_style += "color: #aaa; font-style: italic; "
    base_style += "}"
    combo.setStyleSheet(base_style)
    return combo

class SampledChartDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, color_engine=None, view_combo=None):
        super().__init__(parent)
        self.color_engine = color_engine
        self.view_combo = view_combo

    def sizeHint(self, option, index):
        h = option.rect.height() if option.rect.height() > 0 else 120
        return QSize(int(h * 1.5), h)

    def paint(self, painter, option, index):
        # 1. Fetch the AuditResult object
        result = index.data(Qt.UserRole)
        
        if result and hasattr(result, 'rectified_buffer') and result.rectified_buffer is not None:
            painter.setRenderHint(QPainter.Antialiasing)
            # 2. Identify the "Source" (Audit Display Space) and the "Lens" (Sidebar)
            src_space = result.display_space or "sRGB - Texture"
            tgt_space = self.view_combo.currentText()
            
            # 3. Apply the Bridge Transform (Short-circuits if src == tgt)
            display_buf = self.color_engine.transform_display_bridge(
                result.rectified_buffer, 
                src_space, 
                tgt_space
            )
            
            # 4. Prepare for QImage (Float32 -> UInt8)
            h, w, ch = display_buf.shape
            # Fast vectorized conversion for the UI
            draw_data = (display_buf * 255).astype(np.uint8)
            
            qimg = QImage(draw_data.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # 5. Draw the Pixmap
            rect = option.rect.adjusted(4, 4, -4, -4)
            painter.drawPixmap(rect, pixmap.scaled(
                rect.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))

            # 6. DRAW DYNAMIC OVERLAYS (Neon Green Dots)
            if hasattr(result, 'patch_centers') and result.patch_centers:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 255, 0)) # Neon Green
                
                # We need to scale the coordinates from the buffer size to the UI rect size
                scale_x = rect.width() / w
                scale_y = rect.height() / h
                
                for px, py in result.patch_centers:
                    # Draw a small circle at the scaled coordinate
                    painter.drawEllipse(
                        rect.x() + (px * scale_x) - 2, 
                        rect.y() + (py * scale_y) - 2, 
                        4, 4
                    )

        else:
            painter.setPen(QColor("#666666"))
            painter.drawText(option.rect, Qt.AlignCenter, "No Image")

        
class TrianglePatchDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, color_engine=None, view_combo=None):
        super().__init__(parent)
        self.color_engine = color_engine
        self.view_combo = view_combo

    def sizeHint(self, option, index):
        # We will update the index check later when we reorder columns
        h = option.rect.height() if option.rect.height() > 0 else 120
        padding = 3
        patch_h = (h - (padding * 5)) / 4
        total_w = (patch_h * 6) + (padding * 7)
        return QSize(total_w, h)

    def paint(self, painter, option, index):
        # 1. Grab the result object (stored in Column 6, but we can access via row)
        # It's safer to grab it from the table directly to ensure sync
        result = index.model().data(index.model().index(index.row(), 6), Qt.UserRole)
        if not result or not result.patches:
            return super().paint(painter, option, index)

        # 2. Setup the Lens
        src_space = result.display_space or "sRGB - Texture"
        tgt_space = self.view_combo.currentText()

        rect = option.rect
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        cols, rows = 6, 4
        padding = 3
        patch_size = (rect.height() - (padding * (rows + 1))) / rows
        grid_w = (patch_size * cols) + (padding * (cols - 1))
        offset_x = rect.x() + (rect.width() - grid_w) / 2
        offset_y = rect.y() + padding

        for r in range(rows):
            for c in range(cols):
                patch_idx = r * cols + c
                if patch_idx >= len(result.patches): break
                
                patch = result.patches[patch_idx]
                
                # 3. Transform the Patch Colors through the Bridge
                # We wrap the RGB in a 1x1 pixel buffer for the ColorEngine
                src_rgb = np.array([[patch.visual_src_rgb]], dtype=np.float32)
                ref_rgb = np.array([[patch.visual_ref_rgb]], dtype=np.float32)
                
                view_src = self.color_engine.transform_display_bridge(src_rgb, src_space, tgt_space)
                view_ref = self.color_engine.transform_display_bridge(ref_rgb, src_space, tgt_space)
                
                # 4. Draw Triangles with transformed colors
                x = offset_x + (c * (patch_size + padding))
                y = offset_y + (r * (patch_size + padding))
                patch_rect = QRectF(x, y, patch_size, patch_size)
                
                painter.setPen(Qt.NoPen)
                
                # Top-Left Triangle (Source/Observed)
                s = (view_src[0][0] * 255).astype(np.uint8)
                painter.setBrush(QColor(s[0], s[1], s[2]))
                t1 = QPolygonF([patch_rect.topLeft(), patch_rect.topRight(), patch_rect.bottomLeft()])
                painter.drawPolygon(t1)
                
                # Bottom-Right Triangle (Reference/Target)
                ref = (view_ref[0][0] * 255).astype(np.uint8)
                painter.setBrush(QColor(ref[0], ref[1], ref[2]))
                t2 = QPolygonF([patch_rect.bottomRight(), patch_rect.topRight(), patch_rect.bottomLeft()])
                painter.drawPolygon(t2)
                
        painter.restore()

class ChartMagnifier(QWidget):
    def __init__(self, parent=None, color_engine=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)

        self.setFixedSize(550, 400)
        self.setStyleSheet("background: #1a1a1a; border: 2px solid #FFD700;")
        self.color_engine = color_engine
        self.view_space = "sRGB - Texture"
        self.result = None

    def paintEvent(self, event):
        if not self.result or self.result.rectified_buffer is None: return
        painter = QPainter(self)

        # APPLY THE LENS
        src_space = self.result.display_space or "sRGB - Texture"
        display_buf = self.color_engine.transform_display_bridge(
            self.result.rectified_buffer, src_space, self.view_space
        )
        
        h, w, ch = display_buf.shape
        draw_data = (display_buf * 255).astype(np.uint8)
        qimg = QImage(draw_data.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        img_rect = self.rect().adjusted(10, 10, -10, -40)
        painter.drawPixmap(img_rect, pixmap.scaled(img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if hasattr(self.result, 'patch_centers') and self.result.patch_centers:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 255, 0)) # Neon Green
            
            # Since we used KeepAspectRatio, we calculate the actual scale/offset of the image
            scale = min(img_rect.width() / w, img_rect.height() / h)
            offset_x = img_rect.x() + (img_rect.width() - (w * scale)) / 2
            offset_y = img_rect.y() + (img_rect.height() - (h * scale)) / 2

            for px, py in self.result.patch_centers:
                # Draw circles (Radius 3)
                painter.drawEllipse(QRectF(offset_x + (px * scale) - 3, 
                                           offset_y + (py * scale) - 3, 6, 6))
        painter.setPen(QColor("#CCCCCC"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 0, 0, -10), Qt.AlignHCenter | Qt.AlignBottom, "Sampled Chart (AI Rectified View)")

class GridMagnifier(QWidget):
    def __init__(self, parent=None, color_engine=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setFixedSize(450, 320) 
        self.setStyleSheet("background: #1a1a1a; border: 2px solid #555;")
        self.color_engine = color_engine
        self.view_space = "sRGB - Texture"
        self.result = None

    def paintEvent(self, event):
        if not self.result or not self.result.patches: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Setup the Lens
        src_space = self.result.display_space or "sRGB - Texture"
        
        grid_area = self.rect().adjusted(10, 10, -10, -40)
        cols, rows = 6, 4
        padding = 5
        patch_size = min((grid_area.width() - (padding*7))/6, (grid_area.height() - (padding*5))/4)
        grid_w = (patch_size * cols) + (padding * (cols - 1))
        offset_x = (self.width() - grid_w) / 2

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= len(self.result.patches): break
                patch = self.result.patches[idx]

                # 2. Define the specific rectangle for THIS patch
                x = offset_x + (c * (patch_size + padding))
                y = grid_area.y() + (r * (patch_size + padding))
                patch_rect = QRectF(x, y, patch_size, patch_size) # FIX: Define patch_rect

                # 3. Transform colors through the Bridge
                src_rgb = np.array([[patch.visual_src_rgb]], dtype=np.float32)
                ref_rgb = np.array([[patch.visual_ref_rgb]], dtype=np.float32)
                
                v_src = self.color_engine.transform_display_bridge(src_rgb, src_space, self.view_space)
                v_ref = self.color_engine.transform_display_bridge(ref_rgb, src_space, self.view_space)

                # 4. Draw Triangles
                painter.setPen(Qt.NoPen)
                
                # Left Triangle (Observed)
                s = (v_src[0][0] * 255).astype(np.uint8)
                painter.setBrush(QColor(s[0], s[1], s[2]))
                painter.drawPolygon(QPolygonF([patch_rect.topLeft(), patch_rect.topRight(), patch_rect.bottomLeft()]))
                
                # Right Triangle (Reference)
                r_c = (v_ref[0][0] * 255).astype(np.uint8)
                painter.setBrush(QColor(r_c[0], r_c[1], r_c[2]))
                painter.drawPolygon(QPolygonF([patch_rect.bottomRight(), patch_rect.topRight(), patch_rect.bottomLeft()]))

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


class ActionButtons(QWidget):
    """Encapsulates the Edit, Copy, and Delete buttons for a table row."""
    edit_clicked = Signal()
    copy_clicked = Signal()
    delete_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)

        # 1. Redraw/Edit Bounding Box
        self.edit_btn = QPushButton()
        self.edit_btn.setIcon(qta.icon('fa5s.vector-square', color='#FFD700'))
        self.edit_btn.setToolTip("Manually Redraw Bounding Box")
        self.edit_btn.setFixedSize(30, 30)
        self.edit_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; } "
                                   "QPushButton:hover { background-color: #555; }")
        self.edit_btn.clicked.connect(self.edit_clicked.emit)

        # 2. Copy CDL
        self.copy_btn = QPushButton()
        self.copy_btn.setIcon(qta.icon('fa5s.copy', color='#ADD8E6'))
        self.copy_btn.setToolTip("Copy ASC-CDL values to clipboard")
        self.copy_btn.setFixedSize(30, 30)
        self.copy_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; }"
                                   "QPushButton:hover { background-color: #555; }")
        self.copy_btn.clicked.connect(self.copy_clicked.emit)

        # 3. Delete Row
        self.delete_btn = QPushButton()
        self.delete_btn.setIcon(qta.icon('fa5s.trash-alt', color='#ff6666'))
        self.delete_btn.setToolTip("Remove from Session")
        self.delete_btn.setFixedSize(30, 30)
        self.delete_btn.setStyleSheet("QPushButton { background-color: #444; border-radius: 4px; } "
                                     "QPushButton:hover { background-color: #663333; }")
        self.delete_btn.clicked.connect(self.delete_clicked.emit)

        layout.addWidget(self.edit_btn)
        layout.addWidget(self.copy_btn)
        layout.addWidget(self.delete_btn)
        layout.setAlignment(Qt.AlignCenter)

class ReviewSidebar(QFrame):
    """Refactored Sidebar with 'Funnel' ordering and Status Filtering."""
    view_changed = Signal(str)
    reprocess_requested = Signal()
    tolerance_changed = Signal(float)
    filter_changed = Signal(str)  # New signal
    export_requested = Signal()

    def __init__(self, color_engine, session, parent=None):
        super().__init__(parent)
        self.color_engine = color_engine
        self.session = session
        
        self.setFixedWidth(320)
        self.setStyleSheet("background-color: #2b2b2b; border-left: 1px solid #444;")
        self.layout = QVBoxLayout(self)
        
        self._init_ui()

    def _init_ui(self):
        # --- 1. COLOR PIPELINE ---
        pipeline_group = QGroupBox("COLOR PIPELINE")
        pipe_layout = QVBoxLayout(pipeline_group)

        combo_style = """
            QComboBox {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                color: #EEE;
                font-weight: bold;
            }
            /* Subtle border glow on hover - does not affect arrows */
            QComboBox:hover {
                border: 1px solid #ADD8E6;
            }
            /* This styles the actual dropdown list selection */
            QComboBox QAbstractItemView {
                background-color: #222;
                color: #EEE;
                selection-background-color: #4a90e2;
                selection-color: white;
                outline: 0px;
            }
        """
        
        pipe_layout.addWidget(QLabel("Audit Space (Locked):"))
        self.audit_space_display = QComboBox()
        self.audit_space_display.addItems(["ACEScg"])
        self.audit_space_display.setEnabled(False) 
        self.audit_space_display.setStyleSheet(combo_style + "QComboBox { color: #888; font-style: italic; }")
        pipe_layout.addWidget(self.audit_space_display)

        pipe_layout.addWidget(QLabel("View Transform (UI Only):"))
        self.view_transform_combo = QComboBox()
        src_list, _ = self.color_engine.get_ui_lists()
        self.view_transform_combo.addItems(src_list)
        
        # Set initial value
        first_res = next(iter(self.session.results.values()), None)
        initial_space = first_res.display_space if first_res and first_res.display_space else "sRGB - Texture"
        self.view_transform_combo.setCurrentText(initial_space)
        
        # Apply style with Blue text override
        self.view_transform_combo.setStyleSheet(combo_style + "QComboBox { color: #ADD8E6; }")
        self.view_transform_combo.currentTextChanged.connect(self.view_changed.emit)
        pipe_layout.addWidget(self.view_transform_combo)
        self.layout.addWidget(pipeline_group)

        # --- 2. STATUS FILTER ---
        filter_group = QGroupBox("STATUS FILTER")
        filter_layout = QVBoxLayout(filter_group)
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["SHOW ALL", "PASS", "FAIL (DeltaE)", "FAIL (Geometry)", "DIRTY"])
        self.status_filter_combo.setStyleSheet(combo_style)
        self.status_filter_combo.currentTextChanged.connect(self.filter_changed.emit)
        filter_layout.addWidget(self.status_filter_combo)
        self.layout.addWidget(filter_group)

        # --- 3. AUDIT THRESHOLDS ---
        tol_group = QGroupBox("AUDIT THRESHOLDS")
        tol_layout = QVBoxLayout(tol_group)
        tol_layout.addWidget(QLabel("DeltaE Tolerance:"))
        self.de_spin = QDoubleSpinBox()
        self.de_spin.setRange(0.1, 10.0)
        self.de_spin.setSingleStep(0.1)
        self.de_spin.setDecimals(1)
        self.de_spin.setValue(2.0)
        self.de_spin.valueChanged.connect(self.tolerance_changed.emit)
        tol_layout.addWidget(self.de_spin)
        self.layout.addWidget(tol_group)

        # --- 4. ACTIONS ---
        proc_group = QGroupBox("ACTIONS")
        proc_layout = QVBoxLayout(proc_group)
        self.reprocess_btn = QPushButton(" REPROCESS DIRTY FILES")
        self.reprocess_btn.setIcon(qta.icon('fa5s.sync', color='#FFD700'))
        self.reprocess_btn.setStyleSheet("background-color: #444; font-weight: bold; height: 40px;")
        self.reprocess_btn.clicked.connect(self.reprocess_requested.emit)
        proc_layout.addWidget(self.reprocess_btn)
        self.layout.addWidget(proc_group)

        # Push the remaining items to the bottom
        self.layout.addStretch()

        # --- 5. EXPORT DIRECTORY (Pinned to bottom) ---
        self.export_dir_layout = QVBoxLayout()
        self.layout.addLayout(self.export_dir_layout)

        # --- 6. EXPORT OPTIONS (Pinned to bottom) ---
        exp_group = QGroupBox("EXPORT OPTIONS")
        exp_layout = QVBoxLayout(exp_group)
        self.check_cdl = QCheckBox("Export ASC-CDL (.cdl)")
        self.check_lut = QCheckBox("Export Cube LUT (.cube)")
        self.check_matrix = QCheckBox("Export Matrix (.mtx)")
        self.check_pdf = QCheckBox("Generate PDF Report")
        self.check_csv = QCheckBox("Generate CSV Summary")
        for cb in [self.check_cdl, self.check_lut, self.check_matrix, self.check_pdf, self.check_csv]:
            exp_layout.addWidget(cb)
        self.layout.addWidget(exp_group)

        # --- 7. FINAL EXPORT BUTTON ---
        self.export_btn = QPushButton("EXPORT VALIDATED ITEMS")
        self.export_btn.setMinimumHeight(60)
        self.export_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.layout.addWidget(self.export_btn)