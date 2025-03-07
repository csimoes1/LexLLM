import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb

@dataclass
class TrainingConfigAWS:
    """Configuration for training parameters."""
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    learning_rate: float = 0.0002 # 5e-4
    accumulation_steps: int = 4
    epochs: int = 10  # Increased from 5 to 10
    max_length: int = 256
    batch_size: int = 10
    lora_r: int = 16
    lora_alpha: int = 16 # often set to (2 * lora_r)
    lora_dropout: float = 0.3
    torch_dtype: torch.dtype = torch.bfloat16
    num_workers: int = 2
    output_dir: str = f"lex_lora_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir: str = "transcripts_jsonl"
    patience: int = 2  # New: Number of epochs to wait for val loss improvement

class CustomDataset(Dataset):
    """Dataset class for lazy tokenization of text data."""
    def __init__(self, data, tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Initialized dataset with {len(data)} examples (lazy loading)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = self.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        return {k: v.squeeze(0) for k, v in tokenized.items()}

def collate_fn(batch):
    """Collate function to stack tensors (kept on CPU)."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch])
    }

def setup_environment():
    """Set up CUDA environment variables."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

def load_model_and_tokenizer(config: TrainingConfigAWS) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer with LoRA configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"torch_dtype: {config.torch_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(device)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_data(config: TrainingConfigAWS) -> Dict[str, List[str]]:
    """Load dataset files for training, validation, and testing."""
    def load_files_with_prefix(directory: str, prefix: str, suffix: str = ".jsonl") -> List[str]:
        files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return files

    print("Loading datasets...")
    data_files = {
        "train": load_files_with_prefix(config.data_dir, "lex_episode_training_"),
        "validation": load_files_with_prefix(config.data_dir, "lex_episode_validation_"),
        "test": load_files_with_prefix(config.data_dir, "lex_episode_test_")
    }
    dataset = load_dataset("json", data_files=data_files, streaming=False)
    print(dataset)
    return dataset

def create_data_loaders(
        dataset: Dict[str, List[str]],
        tokenizer: AutoTokenizer,
        config: TrainingConfigAWS
) -> Dict[str, DataLoader]:
    """Create data loaders for each dataset split."""
    loaders = {}
    for split in ["train", "validation", "test"]:
        ds = CustomDataset(dataset[split], tokenizer, config.max_length)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=True
        )
    print("Data loaders created")
    return loaders

def train_epoch(
        model: AutoModelForCausalLM,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfigAWS,
        epoch: int
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    print(f"Epoch {epoch+1}/{config.epochs} - Training")
    device = torch.device("cuda")
    for batch_idx, batch in enumerate(loader):
        print(f"{datetime.now()} Training batch {batch_idx+1}/{len(loader)}")
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type="cuda", dtype=config.torch_dtype):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss / config.accumulation_steps
        loss.backward()

        if (batch_idx + 1) % config.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * config.accumulation_steps
        total_loss += batch_loss
        wandb.log({
            "train/batch_loss": batch_loss,
            "train/step": epoch * len(loader) + batch_idx,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}, Train Batch {batch_idx+1}/{len(loader)}, Loss: {batch_loss:.4f}", end="\r")

    if len(loader) % config.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)

def evaluate_epoch(
        model: AutoModelForCausalLM,
        loader: DataLoader,
        config: TrainingConfigAWS,
        epoch: int,
        phase: str
) -> float:
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss = 0

    print(f"Epoch {epoch+1}/{config.epochs} - {phase.capitalize()}")
    device = torch.device("cuda")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"{datetime.now()} {phase.capitalize()} batch {batch_idx+1}/{len(loader)}")
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type="cuda", dtype=config.torch_dtype):
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
            batch_loss = loss.item()
            total_loss += batch_loss
            wandb.log({
                f"{phase}/batch_loss": batch_loss,
                f"{phase}/step": epoch * len(loader) + batch_idx
            })
            print(f"Epoch {epoch+1}, {phase.capitalize()} Batch {batch_idx+1}/{len(loader)}, Loss: {batch_loss:.4f}", end="\r")

    return total_loss / len(loader)

def save_checkpoint(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: TrainingConfigAWS,
        epoch: Optional[int] = None
):
    """Save model checkpoint or final model."""
    dir_name = f"checkpoint_epoch_{epoch+1}" if epoch is not None else "final"
    checkpoint_dir = os.path.join(config.output_dir, dir_name)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved model to {checkpoint_dir}")

def main():
    """Main training loop with early stopping."""
    config = TrainingConfigAWS()
    setup_environment()

    # Initialize wandb
    wandb.init(project="llama-lora-finetuning", config=config.__dict__)

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = load_data(config)
    loaders = create_data_loaders(dataset, tokenizer, config)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    os.makedirs(config.output_dir, exist_ok=True)
    results_file = os.path.join(config.output_dir, "training_results.json")
    results = {"epochs": []}

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(config.epochs):
        avg_train_loss = train_epoch(model, loaders["train"], optimizer, config, epoch)
        print(f"\nEpoch {epoch+1}, Avg Train Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate_epoch(model, loaders["validation"], config, epoch, "validation")
        print(f"\nEpoch {epoch+1}, Avg Val Loss: {avg_val_loss:.4f}")

        avg_test_loss = evaluate_epoch(model, loaders["test"], config, epoch, "test")
        print(f"\nEpoch {epoch+1}, Avg Test Loss: {avg_test_loss:.4f}")

        epoch_result = {
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_test_loss": avg_test_loss,
            "timestamp": datetime.now().isoformat()
        }
        results["epochs"].append(epoch_result)

        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "test/epoch_loss": avg_test_loss,
            "epoch": epoch + 1
        })

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save the best model checkpoint
            save_checkpoint(model, tokenizer, config, epoch)
            print(f"New best validation loss: {best_val_loss:.4f}, checkpoint saved.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{config.patience}")
            save_checkpoint(model, tokenizer, config, epoch)  # Still save per epoch for reference

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            print(f"Best validation loss: {best_val_loss:.4f} at Epoch {best_epoch}")
            break

        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {results_file}")

    # Save final model (if not stopped early)
    if patience_counter < config.patience:
        print("Saving final model...")
        save_checkpoint(model, tokenizer, config)

    artifact = wandb.Artifact("fine_tuned_lex_llama_lora_AWS", type="model")
    artifact.add_dir(os.path.join(config.output_dir, f"checkpoint_epoch_{best_epoch}"))  # Use best checkpoint
    wandb.log_artifact(artifact)

    print(f"Final Time: {datetime.now()}")
    for epoch_data in results["epochs"]:
        print(f"Epoch {epoch_data['epoch']}, Avg Train Loss: {epoch_data['avg_train_loss']:.4f}, "
              f"Avg Val Loss: {epoch_data['avg_val_loss']:.4f}, Avg Test Loss: {epoch_data['avg_test_loss']:.4f}, "
              f"Timestamp: {epoch_data['timestamp']}")

    wandb.finish()

if __name__ == "__main__":
    main()