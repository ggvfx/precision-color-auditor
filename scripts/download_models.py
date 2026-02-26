"""
Precision Color Auditor - Model Hydration Script
Downloads the Florence-2 model weights to the local resources folder.
Run this script once after cloning the repository.
"""

import os
from huggingface_hub import snapshot_download

def hydrate_models():
    # Define the target directory relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "src", "resources", "models", "florence2")

    print(f"[*] Initializing model download to: {target_dir}")
    
    try:
        snapshot_download(
            repo_id="microsoft/Florence-2-base",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print("[+] Model hydration complete.")
    except Exception as e:
        print(f"[!] Error downloading models: {e}")

if __name__ == "__main__":
    hydrate_models()