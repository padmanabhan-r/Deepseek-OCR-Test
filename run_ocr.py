import os
import warnings
import sys

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
sys.stderr = open(os.devnull, 'w')

import torch
from transformers import AutoModel, AutoTokenizer

sys.stderr = sys.__stderr__

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

# USE LOCAL DIRECTORY - this is critical!
MODEL_DIR = "./DeepSeek-OCR"

print(f"Using device: {device}")
print(f"Loading from: {MODEL_DIR}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    attn_implementation="eager",
)

# Store device on model so patched code can access it
model = model.to(device=device, dtype=dtype).eval()

print("Running inference...")
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = "sample.jpg"
output_path = "out"
os.makedirs(output_path, exist_ok=True)

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024, image_size=640, crop_mode=True,
    save_results=True, test_compress=True,
    device=device  # <- ADD THIS LINE
)

print("\n=== OCR OUTPUT ===\n", res)
print("\nSaved artifacts in:", output_path)