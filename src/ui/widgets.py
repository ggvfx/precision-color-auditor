from PySide6.QtWidgets import (QComboBox, QWidget, QStyledItemDelegate)
from PySide6.QtCore import Qt, QSize, QRectF, QRect
from PySide6.QtGui import QColor, QPainter, QPolygonF, QImage, QPixmap
import numpy as np

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