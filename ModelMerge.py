from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# Paths
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
working_dir = "/Users/csimoes/Projects/llama/final_Mar052024"
lora_model_path = working_dir
merged_model_path = working_dir

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load LoRA model
model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge LoRA adapters into base model
model = model.merge_and_unload()  # Combines LoRA weights into base model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# Save merged model
os.makedirs(merged_model_path, exist_ok=True)
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved to {merged_model_path}")