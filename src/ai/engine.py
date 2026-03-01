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
        """Extracts coordinates from raw Florence-2 location tokens."""
        if isinstance(roi_result, list):
            roi_result = roi_result[0]

        # 1. Get the raw string from the dictionary
        raw_str = ""
        for val in roi_result.values():
            if isinstance(val, str):
                raw_str = val
                break

        # 2. Use Regex to find all <loc_###> patterns
        tokens = re.findall(r'<loc_(\d+)>', raw_str)
        
        if len(tokens) >= 4:
            # Florence-2 uses a 1000x1000 coordinate system
            # Format is usually [ymin, xmin, ymax, xmax]
            coords = [int(t) / 1000.0 for t in tokens[:4]]
            
            # We need the original image dimensions to scale back
            # These should be available from your engine's state or passed in
            # For now, let's create the 4 corners of the bounding box
            ymin, xmin, ymax, xmax = coords
            
            # Convert to 4 corners for the Locator: [top-left, top-right, bottom-right, bottom-left]
            # We multiply by image size in the Locator, so we return normalized 0-1 points here
            poly = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax]
            ])
            return poly

        return []