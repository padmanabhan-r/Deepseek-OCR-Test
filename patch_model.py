import os
import sys

# Download model first
print("Downloading model...")
from huggingface_hub import snapshot_download
model_path = snapshot_download("deepseek-ai/DeepSeek-OCR", local_dir="./DeepSeek-OCR")

# Patch the modeling file to support MPS
modeling_file = os.path.join(model_path, "modeling_deepseekocr.py")

print(f"Patching {modeling_file} for MPS support...")

with open(modeling_file, 'r') as f:
    content = f.read()

# Replace all .cuda() calls with .to(device)
replacements = [
    ('.cuda()', '.to(self.device)'),
    ('torch.cuda.', 'torch.'),
    ("device='cuda'", "device=self.device"),
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open(modeling_file, 'w') as f:
    f.write(content)

print("✓ Model patched successfully!")
print("Now you can run inference with MPS support")