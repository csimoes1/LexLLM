from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import torch
from torch.amp import autocast
import os
import wandb
import json  # For saving results to a file

#get datetime string for use in filenames
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize results as a dictionary for structured storage
results = {"epochs": []}
model_id = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "lex_lora_results"+datetime_str

learning_rate = 5e-4
accumulation_steps = 4
epochs = 3
max_length = 256
batch_size = 4
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

# Initialize wandb
wandb.init(
    project="llama-lora-finetuning",
    config={
        "model_id": model_id,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "accumulation_steps": accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "max_length": max_length
    }
)

# Set environment variable for MPS optimization
os.environ["PYTORCH_MPS_HIGH_WATERMARK"] = "0.8"

# Load pre-trained model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
torch_dtype = torch.float16
print(f"torch_dtype: {torch_dtype}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)

# Add LoRA configuration
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Pre-tokenize dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("Pre-tokenizing dataset...")
        self.inputs = [
            tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            ) for item in data
        ]
        self.inputs = [{k: v.squeeze(0).to(device) for k, v in input_dict.items()} for input_dict in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch])
    }

def load_files_with_prefix(directory, prefix, suffix=".jsonl"):
    result = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            result.append(directory+"/"+filename)
    # sort result by episode number
    # example episode filename: lex_episode_training_1.jsonl
    result.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return result

# Load training and testing data
print("Loading datasets...")
train_files = load_files_with_prefix("transcripts_jsonl", "lex_episode_training_")
validation_files = load_files_with_prefix("transcripts_jsonl", "lex_episode_validation_")
test_files = load_files_with_prefix("transcripts_jsonl", "lex_episode_test_")

data_files = {
    "train": train_files,
    "validation": validation_files,
    "test": test_files
}
dataset = load_dataset("json", data_files=data_files)
print(dataset)

# Check token lengths
for split in ["train", "validation", "test"]:
    j = 0
    for i, item in enumerate(dataset[split]):
        tokens = tokenizer(item["text"], return_tensors="pt")["input_ids"][0]
        if len(tokens) > 256:
            print(f"{split.capitalize()} Example {i}: {len(tokens)} tokens")
            j += 1
    print(f"Total {split} examples={len(dataset[split])}, examples with more than 256 tokens: {j}")

# Create datasets and data loaders
train_dataset = CustomDataset(dataset["train"], tokenizer)
validation_dataset = CustomDataset(dataset["validation"], tokenizer)
test_dataset = CustomDataset(dataset["test"], tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)
print("Data loaders created")

# Training and evaluation loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Directory for saving results and checkpoints
os.makedirs(output_dir, exist_ok=True)
results_file = os.path.join(output_dir, "training_results.json")

print("Starting training...")
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{epochs} - Training")
    for batch_idx, batch in enumerate(train_loader):
        print(f"{datetime.now()} Training batch {batch_idx+1}/{len(train_loader)}")
        with autocast(device_type="mps"):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * accumulation_steps
        total_train_loss += batch_loss
        wandb.log({
            "train/batch_loss": batch_loss,
            "train/step": epoch * len(train_loader) + batch_idx,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}, Train Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss:.4f}", end="\r")

    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"\nEpoch {epoch+1}, Avg Train Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{epochs} - Validation")
        for batch_idx, batch in enumerate(validation_loader):
            print(f"{datetime.now()} Validation batch {batch_idx+1}/{len(validation_loader)}")
            with autocast(device_type="mps"):
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
            batch_loss = loss.item()
            total_val_loss += batch_loss
            wandb.log({
                "val/batch_loss": batch_loss,
                "val/step": epoch * len(validation_loader) + batch_idx
            })
            print(f"Epoch {epoch+1}, Val Batch {batch_idx+1}/{len(validation_loader)}, Loss: {batch_loss:.4f}", end="\r")

    avg_val_loss = total_val_loss / len(validation_loader)
    print(f"\nEpoch {epoch+1}, Avg Val Loss: {avg_val_loss:.4f}")

    # Testing phase
    total_test_loss = 0
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{epochs} - Testing")
        for batch_idx, batch in enumerate(test_loader):
            print(f"{datetime.now()} Testing batch {batch_idx+1}/{len(test_loader)}")
            with autocast(device_type="mps"):
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
            batch_loss = loss.item()
            total_test_loss += batch_loss
            wandb.log({
                "test/batch_loss": batch_loss,
                "test/step": epoch * len(test_loader) + batch_idx
            })
            print(f"Epoch {epoch+1}, Test Batch {batch_idx+1}/{len(test_loader)}, Loss: {batch_loss:.4f}", end="\r")

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\nEpoch {epoch+1}, Avg Test Loss: {avg_test_loss:.4f}")

    # Store epoch results
    epoch_result = {
        "epoch": epoch + 1,
        "avg_train_loss": avg_train_loss,
        "avg_test_loss": avg_test_loss,
        "timestamp": datetime.now().isoformat()
    }
    results["epochs"].append(epoch_result)

    # Log epoch-level metrics to W&B
    wandb.log({
        "train/epoch_loss": avg_train_loss,
        "val/epoch_loss": avg_val_loss,
        "test/epoch_loss": avg_test_loss,
        "epoch": epoch + 1
    })

    # Save results to JSON file after each epoch
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

    # Save model checkpoint after each epoch
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved model checkpoint to {checkpoint_dir}")

# Save final model
print("Saving final model...")
final_model_dir = os.path.join(output_dir, "final")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Log final model artifacts to W&B
artifact = wandb.Artifact("fine_tuned_lex_llama_lora", type="model")
artifact.add_dir(final_model_dir)
wandb.log_artifact(artifact)

print(f"Final Time: {datetime.now()}")

# Print all results
for epoch_data in results["epochs"]:
    print(f"Epoch {epoch_data['epoch']}, Avg Train Loss: {epoch_data['avg_train_loss']:.4f}, "
          f"Avg Test Loss: {epoch_data['avg_test_loss']:.4f}, Timestamp: {epoch_data['timestamp']}")

# Finish W&B run
wandb.finish()