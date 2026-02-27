"""
Precision Color Auditor - Smart Patch Discovery
Uses AI-driven region proposals to locate individual color patches 
without hardcoded grid templates.
"""

import numpy as np
from typing import List, Tuple, Dict
from core.models import ColorPatch
from core.config import settings

class PatchSampler:
    """
    Analyzes an ROI to discover, sort, and sample individual color patches.
    """

    def __init__(self, detector):
        self.detector = detector  # Pass the ChartDetector instance to use Florence-2

    def discover_and_sample(self, image_buffer: np.ndarray, chart_roi: dict) -> List[ColorPatch]:
        """
        The primary discovery loop: Proposals -> Filtering -> Sorting -> Sampling.
        """
        # 1. Get Proposals from the AI (Florence-2 <REGION_PROPOSAL>)
        # This replaces hardcoded grids with visual discovery.
        raw_proposals = self._get_patch_proposals(image_buffer, chart_roi)
        
        # 2. Filter and Sort the patches into a 2D grid
        sorted_coords = self._sort_patches(raw_proposals)
        
        # 3. Create ColorPatch objects by sampling the image buffer
        sampled_patches = []
        for i, coords in enumerate(sorted_coords):
            rgb = self._sample_mean_rgb(image_buffer, coords)
            
            # Note: target_rgb is TBD until the Auditor identifies the chart signature
            patch = ColorPatch(
                name=f"Patch_{i}",
                observed_rgb=rgb,
                target_rgb=np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                coordinates=coords,
                index=i
            )
            sampled_patches.append(patch)
            
        return sampled_patches

    def _get_patch_proposals(self, image_buffer: np.ndarray, roi: dict) -> List[Tuple[int, int, int, int]]:
        """
        Uses Florence-2 to find individual patches within the cropped chart ROI.
        """
        # 1. Extract the bounding box from the detector's ROI format
        # Florence-2 output from detector is usually {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[...]], 'labels': [...]}}
        task_key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if task_key not in roi or not roi[task_key]['bboxes']:
            return []
        
        # We take the first detected chart box
        chart_bbox = roi[task_key]['bboxes'][0] # [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = map(int, chart_bbox)

        # 2. Crop the image to the chart area for higher detection precision
        crop = image_buffer[ymin:ymax, xmin:xmax]
        pil_crop = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))

        # 3. Run Florence-2 Region Proposal on the crop
        task_prompt = "<REGION_PROPOSAL>"
        inputs = self.detector.processor(
            text=task_prompt, 
            images=pil_crop, 
            return_tensors="pt"
        ).to(self.detector.device)

        import torch
        with torch.no_grad():
            generated_ids = self.detector.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )

        # 4. Parse the results
        results = self.detector.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.detector.processor.post_process_generation(
            results, 
            task=task_prompt, 
            image_size=(pil_crop.width, pil_crop.height)
        )

        # 5. Map the patch coordinates back to the original image space
        proposals = []
        for patch_bbox in parsed[task_prompt]['bboxes']:
            p_ymin, p_xmin, p_ymax, p_xmax = patch_bbox
            # Offset by the original crop coordinates
            proposals.append((
                p_ymin + ymin, 
                p_xmin + xmin, 
                p_ymax + ymin, 
                p_xmax + xmin
            ))

        return proposals

    def _sort_patches(self, proposals: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Sorts the discovered patches spatially (Row-major order).
        This handles charts that are rotated or captured at slight angles.
        """
        if not proposals:
            return []
            
        # Sort by Y (top to bottom) then X (left to right)
        # We add a 'tolerance' to Y so patches in the same row stay together
        avg_height = np.mean([p[2] - p[0] for p in proposals])
        proposals.sort(key=lambda p: (p[0] // (avg_height * 0.5), p[1]))
        
        return proposals

    def _sample_mean_rgb(self, buffer: np.ndarray, coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Samples the mean color of a patch using the global sample_size (e.g., 32x32).
        Calculates the center of the coordinate box to avoid sampling edges/borders.
        """
        ymin, xmin, ymax, xmax = coords
        center_y, center_x = (ymin + ymax) // 2, (xmin + xmax) // 2
        
        half_size = settings.sample_size // 2
        
        # Define the sample window
        y_start, y_end = int(center_y - half_size), int(center_y + half_size)
        x_start, x_end = int(center_x - half_size), int(center_x + half_size)
        
        # Slice the high-precision buffer and calculate the mean
        sample_area = buffer[y_start:y_end, x_start:x_end]
        return np.mean(sample_area, axis=(0, 1))