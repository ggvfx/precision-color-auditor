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

    def detect_with_fallback(self, display_buffer, audit_buffer):
        """
        Tries to find the chart using the display buffer first.
        If it fails, it applies a safety gamma to the audit buffer.
        """
        # Pass 1: The User's Display Space
        result, reasoning = self.detect_chart_roi(display_buffer)
        
        # --- NEW SMARTER SUCCESS CHECK ---
        is_success = False
        data = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
        bboxes = data.get("bboxes", [])
        
        if bboxes:
            box = bboxes[0] # [xmin, ymin, xmax, ymax]
            # Calculate how much of the image this box covers (0-1000 scale)
            # A chart almost never fills 95% of the frame in an audit.
            width_norm = box[2] - box[0]
            height_norm = box[3] - box[1]
            coverage = (width_norm * height_norm) / (1000 * 1000)
            
            # If box is valid AND not just the full frame hallucination
            if coverage < 0.95:
                is_success = True

        if is_success:
            return result, reasoning
            
        # Pass 2: Safety Fallback
        print("[AI] Display space detection failed or weak (Full-frame hallucination). Attempting Safety Gamma...")
        
        # Apply the 2.2 gamma shift to the Linear ACEScg buffer
        safety_buffer = np.power(np.clip(audit_buffer, 0, 1), 1/2.2)
        
        res_fb, reason_fb = self.detect_chart_roi(safety_buffer)
        return res_fb, f"[FALLBACK-MODE] {reason_fb}"

    def detect_chart_roi(self, image_array: np.ndarray) -> tuple[dict, str]:
        """
        Runs Florence-2 detection and returns both the parsed coordinates 
        and the raw reasoning string for the UI logs.
        """
        height, width = image_array.shape[:2]
        
        if image_array.dtype == np.uint8:
            pil_img = Image.fromarray(image_array)
        else:
            pil_img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        
        template = settings.get_current_template()
        description = template.detection_prompt

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
        
        # 1. Capture the raw string
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # debug print
        print(f"\n[RAW AI OUTPUT - GROUNDING] {results}")
        
        # 2. Parse the answer
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=task_tag, 
            image_size=(width, height)
        )
        
        # 3. Return both as a tuple
        return parsed_answer, results

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