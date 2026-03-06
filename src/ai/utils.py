"""
AI Utilities - Image Processing Handshakes
-----------------------------------------
Handles the conversion between high-precision float buffers (0-1) 
and the integer-based formats (uint8) required by Pillow and Florence-2.
"""

import numpy as np
from PIL import Image

def prep_for_pil(buffer: np.ndarray) -> Image.Image:
    """
    Optimized conversion from float32/float16 buffers to uint8 PIL Images.
    Used for high-frequency UI previews and AI detection.
    """
    # 1. Fast Path: If already uint8 (0-255), just wrap it as a PIL image
    if buffer.dtype == np.uint8:
        return Image.fromarray(buffer)
    
    # 2. Performance Path: 
    # Multiply and cast in a single expression to minimize intermediate copies.
    # We use 255.0 to maintain precision before the integer floor.
    img_data = (np.clip(buffer, 0, 1) * 255.0).astype(np.uint8)
    
    return Image.fromarray(img_data)

def normalize_for_ai(buffer: np.ndarray) -> np.ndarray:
    """
    Standardizes a buffer for AI feature detection.
    Stretches contrast based on image percentiles to aid detection in 
    underexposed or log-encoded images.
    """
    # Use a slightly faster approximation or keep percentiles for accuracy
    # 2nd and 98th percentile helps ignore hot pixels/noise
    p2, p98 = np.percentile(buffer, (2, 98))
    
    span = p98 - p2
    if span < 1e-5:
        return np.clip(buffer, 0, 1)
        
    # In-place math to save memory
    normalized = buffer - p2
    normalized /= span
    return np.clip(normalized, 0, 1)

def get_bytes_size(buffer: np.ndarray) -> str:
    """Utility to track memory usage of EXR buffers in the session."""
    size_bytes = buffer.nbytes
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"