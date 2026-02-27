import json
import os

# Using the path we verified
path = r'D:/_repos/precision-color-auditor/src/resources/models/florence2/tokenizer.json'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

vocab = data["model"]["vocab"]
current_size = len(vocab)
target_size = 51289

if current_size < target_size:
    print(f"Repairing tokenizer: {current_size} -> {target_size}")
    # Florence-2 coordinate tokens are added as strings: <0> to <999>
    # Plus specialized task tokens
    for i in range(current_size, target_size):
        vocab[f"<extra_id_{i-current_size}>"] = i
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("Repair Complete.")
else:
    print(f"Tokenizer already has {current_size} tokens. No repair needed.")