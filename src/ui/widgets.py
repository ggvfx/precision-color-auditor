from PySide6.QtWidgets import QComboBox
from PySide6.QtCore import Qt

def create_ocio_combo(color_engine, current_value=None, is_fallback=False):
    """
    Standardized factory for OCIO input space dropdowns used across the app.
    """
    combo = QComboBox()
    
    # Get the list once
    src_list, _ = color_engine.get_ui_lists()
    combo.addItems(src_list)
    
    if current_value:
        # If the value isn't in our list (e.g. from a broken sidecar), add it temporarily
        if current_value not in src_list:
            combo.insertItem(0, current_value)
        combo.setCurrentText(current_value)

    # Standardized Styling
    base_style = "QComboBox { background-color: #333; padding: 2px; "
    if is_fallback:
        base_style += "color: #aaa; font-style: italic; "
    base_style += "}"
    
    combo.setStyleSheet(base_style)
    return combo