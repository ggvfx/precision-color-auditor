import re
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
        height, width = image_array.shape[:2]
        
        if image_array.dtype == np.uint8:
            pil_img = Image.fromarray(image_array)
        else:
            # Apply a 2.2 gamma lift to "linear" data so the AI can actually see the chart
            normalized = np.clip(image_array, 0, 1)
            gamma_lifted = np.power(normalized, 1/2.2) 
            pil_img = Image.fromarray((gamma_lifted * 255).astype(np.uint8))
        
        # Prompt
        current_template = settings.get_current_template()
        description = current_template.get("detection_prompt", "")

        task_tag = "<CAPTION_TO_PHRASE_GROUNDING>"
        full_prompt = f"{task_tag}{description}"
        
        inputs = self.processor(text=full_prompt, images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"\n[RAW AI OUTPUT - GROUNDING] {results}")
        
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=task_tag, 
            image_size=(width, height)
        )
        
        return parsed_answer

    def extract_polygons(self, roi_result, width: int, height: int) -> np.ndarray:
        if isinstance(roi_result, list):
            roi_result = roi_result[0]

        # 1. Try Grounding Result
        data = roi_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        if isinstance(data, dict) and 'bboxes' in data:
            bboxes = data.get('bboxes', [])
            if bboxes:
                xmin, ymin, xmax, ymax = bboxes[0]
                return np.array([
                    [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                ])

        # 2. Fallback to raw token parsing
        raw_str = str(roi_result)
        tokens = re.findall(r'<loc_(\d+)>', raw_str)
        if len(tokens) >= 4:
            vals = [int(t) / 1000.0 for t in tokens]
            ymin, xmin, ymax, xmax = vals[:4]
            return np.array([
                [xmin * width, ymin * height], [xmax * width, ymin * height], 
                [xmax * width, ymax * height], [xmin * width, ymax * height]
            ])

        return np.array([])