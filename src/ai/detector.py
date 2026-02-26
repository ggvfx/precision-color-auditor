"""
Precision Color Auditor - AI-Driven Chart Localization
Wraps Florence-2 for automated Macbeth and Greyscale chart detection.
Maps relative ROI coordinates for high-precision pixel sampling.
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from core.config import settings

class ChartDetector:
    """
    Handles local AI inference to locate color references within an image.
    """

    def __init__(self):
        # Local inference ensures standalone integrity 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Point to the local directory where weights were hydrated
        self.model_path = os.path.join(
            settings.app_root, "src", "resources", "models", "florence2"
        )
        
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Initializes Florence-2 from the local filesystem."""
        try:
            # We load from self.model_path instead of a web-based ID
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            ).to(self.device).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Local AI Model Load Failed. Ensure weights exist in {self.model_path}. Error: {e}"
            )

    def detect_chart_roi(self, image_array: np.ndarray) -> dict:
        """
        Locates the Macbeth or Greyscale chart ROI in a 'noisy' environment.
        
        Args:
            image_array: The image as a float32 NumPy array.
            
        Returns:
            dict: Bounding box coordinates for the detected chart.
        """
        # Convert to 8-bit for the vision model's processing
        pil_img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        
        prompt = "<OD>" 
        text_input = "color checker chart"
        
        inputs = self.processor(text=prompt + text_input, images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=prompt, 
            image_size=(pil_img.width, pil_img.height)
        )

        return parsed_answer # Provides the Region of Interest (ROI)