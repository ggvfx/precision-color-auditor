import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from core.config import settings

class ChartDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "microsoft/Florence-2-base"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        print(f"[DEBUG] Loading {self.model_id} from cache...")
        # We keep trust_remote_code=True to use the official MS scripts you just downloaded
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        print(f"[SUCCESS] Florence-2 ready on {self.device}")

    def detect_chart_roi(self, image_array: np.ndarray) -> dict:
        # 1. Image prep
        pil_img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        
        # 2. Set the task to Grounding
        task_tag = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = "color calibration chart"
        full_prompt = f"{task_tag}{text_input}"
        
        inputs = self.processor(text=full_prompt, images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # 3. Inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # 4. Parse coordinates
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=task_tag, 
            image_size=(pil_img.width, pil_img.height)
        )
        
        print(f"[DEBUG] Found: {parsed_answer}")
        return parsed_answer