import os
import re

modeling_file = "./DeepSeek-OCR/modeling_deepseekocr.py"

print(f"Patching {modeling_file} for MPS support...")

with open(modeling_file, 'r') as f:
    content = f.read()

# Fix the infer method signature - set default to string "cuda"
infer_pattern = r'def infer\(self, tokenizer, prompt=\'\', image_file=\'\', output_path = \'\', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, device=device\):'

content = content.replace(
    'def infer(self, tokenizer, prompt=\'\', image_file=\'\', output_path = \'\', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, device=device):',
    'def infer(self, tokenizer, prompt=\'\', image_file=\'\', output_path = \'\', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, device="cuda"):'
)

# Replace .cuda() with .to(device)
content = content.replace('.cuda()', '.to(device)')

# Replace torch.cuda. with torch.
content = content.replace('torch.cuda.', 'torch.')

# Replace device='cuda' or device="cuda" with device=device (but not in function signature)
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'def infer(' not in line:
        line = line.replace("device='cuda'", 'device=device')
        line = line.replace('device="cuda"', 'device=device')
    new_lines.append(line)

content = '\n'.join(new_lines)

with open(modeling_file, 'w') as f:
    f.write(content)

print("✓ Model patched successfully!")