from PySide6.QtWidgets import (QComboBox, QWidget, QStyledItemDelegate)
from PySide6.QtCore import Qt, QSize, QRectF, QRect
from PySide6.QtGui import QColor, QPainter, QPolygonF, QImage, QPixmap

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
    def sizeHint(self, option, index):
        h = option.rect.height() if option.rect.height() > 0 else 120
        return QSize(int(h * 1.5), h)

    def paint(self, painter, option, index):
        # Shifted everything left to remove the dependency on the commented-out 'if'
        result = index.data(Qt.UserRole)
        if result and hasattr(result, 'rectified_buffer') and result.rectified_buffer is not None:
            arr = result.rectified_buffer
            h, w, ch = arr.shape
            qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            rect = option.rect.adjusted(4, 4, -4, -4)
            painter.drawPixmap(rect, pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            painter.setPen(QColor("#666666"))
            painter.drawText(option.rect, Qt.AlignCenter, "No Image")

class TrianglePatchDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        # We will update the index check later when we reorder columns
        h = option.rect.height() if option.rect.height() > 0 else 120
        padding = 3
        patch_h = (h - (padding * 5)) / 4
        total_w = (patch_h * 6) + (padding * 7)
        return QSize(total_w, h)

    def paint(self, painter, option, index):
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

class ChartMagnifier(QWidget):
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
        painter.setPen(QColor("#CCCCCC"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 0, 0, -10), Qt.AlignHCenter | Qt.AlignBottom, "Sampled Chart (AI Rectified View)")

class GridMagnifier(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setFixedSize(450, 320) 
        self.setStyleSheet("background: #1a1a1a; border: 2px solid #555;")
        self.result = None

    def paintEvent(self, event):
        if not self.result: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        grid_area = self.rect().adjusted(10, 10, -10, -40)
        cols, rows = 6, 4
        padding = 5
        patch_size = min((grid_area.width() - (padding*7))/6, (grid_area.height() - (padding*5))/4)
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