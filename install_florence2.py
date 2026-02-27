import os
import torch

# FORCE ONLINE MODE AT THE CORE LEVEL
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "microsoft/Florence-2-base"

print(f"--- Starting Forced Online Download of {model_id} ---")

try:
    # We add local_files_only=False to double-down on the instruction
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        local_files_only=False
    ).to("cuda").eval()

    processor = AutoProcessor.from_pretrained(
        model_id, 
        trust_remote_code=True,
        local_files_only=False
    )
    print("--- SUCCESS: Model downloaded and verified! ---")
    
except Exception as e:
    print(f"--- FAILURE: {e} ---")

# pip install --force-reinstall transformers==4.48.3