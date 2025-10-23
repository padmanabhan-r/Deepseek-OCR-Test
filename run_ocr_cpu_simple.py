import os
import torch
from transformers import AutoModel, AutoTokenizer

# Disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    use_safetensors=True,
    attn_implementation='eager'
)

# Convert to float32 for CPU and set to eval
model = model.eval().to(torch.float32)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'sample.jpg'
output_path = 'out'
os.makedirs(output_path, exist_ok=True)

print("Running inference (this will be slow on CPU)...")

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    save_results=True, 
    test_compress=True
)

print("\n=== OCR OUTPUT ===")
print(res)
print(f"\nSaved artifacts in: {output_path}")