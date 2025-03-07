import json
from transformers import AutoTokenizer
import argparse

def count_tokens_in_file(file_path: str, model_id: str = "meta-llama/Llama-3.2-3B-Instruct", max_length: int = None) -> int:
    """
    Count the total number of tokens in a JSONL file using a specified tokenizer.

    Args:
        file_path (str): Path to the JSONL file where each line has a 'text' field.
        model_id (str): Hugging Face model ID for the tokenizer (default: LLaMA 3.2 3B Instruct).
        max_length (int, optional): Maximum sequence length for tokenization. If None, no truncation.

    Returns:
        int: Total number of tokens across all lines in the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If a line isn’t valid JSON.
        KeyError: If a line’s JSON lacks a 'text' field.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded tokenizer for {model_id}")

    # Initialize token count
    total_tokens = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                # Parse JSON line
                try:
                    data = json.loads(line.strip())
                    text = data["text"]
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_idx} due to JSON error: {e}")
                    continue
                except KeyError:
                    print(f"Warning: Skipping line {line_idx} - No 'text' field found")
                    continue

                # Tokenize the text
                tokenized = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True if max_length is not None else False,
                    max_length=max_length if max_length is not None else None,
                    padding=False  # No padding needed for counting
                )
                num_tokens = tokenized["input_ids"].shape[1]  # Number of tokens in this line
                total_tokens += num_tokens
                print(f"Line {line_idx}: {num_tokens} tokens (running total: {total_tokens})")

        print(f"Total tokens in {file_path}: {total_tokens}")
        return total_tokens

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        raise

def main():
    """Command-line interface for counting tokens in a file."""
    parser = argparse.ArgumentParser(description="Count tokens in a JSONL file.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Hugging Face model ID for the tokenizer")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Maximum sequence length (optional, defaults to no truncation)")
    args = parser.parse_args()

    # Count tokens
    total_tokens = count_tokens_in_file(
        file_path=args.file_path,
        model_id=args.model_id,
        max_length=args.max_length
    )
    print(f"Final count: {total_tokens} tokens")

if __name__ == "__main__":
    main()