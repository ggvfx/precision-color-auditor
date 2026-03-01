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
    Safely converts a NumPy float32 color buffer to a PIL Image.
    Handles the 0.0-1.0 to 0-255 scaling.
    """
    # Create a copy to avoid mutating the original buffer in memory
    img_data = np.copy(buffer)
    
    # If the data is float, scale it up
    if img_data.dtype.kind == 'f':
        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
    else:
        img_data = img_data.astype(np.uint8)
        
    return Image.fromarray(img_data)

def normalize_for_ai(buffer: np.ndarray) -> np.ndarray:
    """
    Standardizes a buffer for better AI feature detection.
    Stretches contrast to help find gray patches in low light.
    """
    p2, p98 = np.percentile(buffer, (2, 98))
    # Avoid division by zero
    denom = (p98 - p2) if (p98 - p2) > 1e-5 else 1.0
    return np.clip((buffer - p2) / denom, 0, 1)