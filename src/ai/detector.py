"""
Precision Color Auditor - AI-Driven Chart Localization
Wraps Florence-2 for automated detection of various calibration charts.
Uses the global settings for dynamic, config-driven search prompts.
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
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, Florence2Processor

        state_dict = load_file(os.path.join(self.model_path, "model.safetensors"))
        shared = state_dict["language_model.model.shared.weight"]
        
        state_dict["language_model.model.encoder.embed_tokens.weight"] = shared
        state_dict["language_model.model.decoder.embed_tokens.weight"] = shared
        state_dict["language_model.lm_head.weight"] = shared

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config.vocab_size = 51289
        config.text_config.vocab_size = 51289
        
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        self.processor = Florence2Processor.from_pretrained(self.model_path, local_files_only=True)

    def detect_chart_roi(self, image_array: np.ndarray) -> dict:
        """
        Locates the active chart type defined in the global settings.
        
        Args:
            image_array: The image as a float32 NumPy array.
            
        Returns:
            dict: Bounding box coordinates for the detected chart.
        """
        # 1. Convert 32-bit linear float to 8-bit PIL for the vision model
        pil_img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        
        # 2. Discovery prompt
        prompt = "<CAPTION_TO_PHRASE_GROUNDING>" 
        text_input = "color calibration chart"
        
        # 3. Prepare inputs for local inference
        inputs = self.processor(
            text=prompt + text_input, 
            images=pil_img, 
            return_tensors="pt"
        ).to(self.device)

        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        
        # 4. Generate the detection coordinates
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # 5. Map internal tokens back to image pixel coordinates
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=prompt, 
            image_size=(pil_img.width, pil_img.height)
        )

        return parsed_answer