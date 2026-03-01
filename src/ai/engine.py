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
        """
        Detects the primary chart area and returns normalized and absolute coordinates.
        """
        # 1. Image prep - Ensure we know the original dimensions
        height, width = image_array.shape[:2]
        if image_array.dtype == np.uint8:
            pil_img = Image.fromarray(image_array)
        else:
            pil_img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8))
        
        # 2. Set the task to Polygon Segmentation
        task_tag = "<POLYGON_SEGMENTATION>"
        full_prompt = f"{task_tag}color calibration chart"
        
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
        
        # 4. Parse coordinates - Scaling back to original image size
        parsed_answer = self.processor.post_process_generation(
            results, 
            task=task_tag, 
            image_size=(width, height)
        )
        
        # Note: Polygons use a different key than bboxes in the output
        print(f"[DEBUG] Found Polygon points for ROI.")
        return parsed_answer

    def get_absolute_bbox(self, roi_result) -> list:
        """Extracts the polygon and converts it to 4 corners."""
        key = "<POLYGON_SEGMENTATION>"
        
        # 1. Handle the list wrapping
        if isinstance(roi_result, list) and len(roi_result) > 0:
            roi_result = roi_result[0]
            
        # 2. Safety check: Ensure roi_result is actually a dictionary
        if not isinstance(roi_result, dict):
            print(f"[ERROR] Engine: roi_result is {type(roi_result)}, expected dict.")
            return []

        # 3. Extract the polygon data - Look in EVERY key
        for k, v in roi_result.items():
            if isinstance(v, dict) and 'polygons' in v:
                polygons = v['polygons']
                if polygons and len(polygons) > 0:
                    # Florence returns points as [[[x1, y1, x2, y2...]]]
                    # Flatten the first polygon found and reshape to (N, 2)
                    poly = np.array(polygons[0]).reshape(-1, 2)
                    return poly
        
        # If the loop finishes without returning, we found nothing
        print(f"[DEBUG] Engine: No 'polygons' found in any key. Keys present: {list(roi_result.keys())}")
        return []