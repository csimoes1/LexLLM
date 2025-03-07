from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Full text from your example
text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>would be an actual AI agent that actually goes fully autonomous. Yeah, what if you really set one of these things loose and let it basically build itself a business? And so, I will say, we're not yet seeing those, and I think there's a little bit of the systems aren't quite ready for that yet, and then I think it's a little bit of, you really do need, at that point, a founder who's really willing to break all the rules, and really willing to take the swing, and those people exist, and so I'm sure we'll see that.<|eot_id|><|start_header_id|>assistant<|end_header_id|>And some of it is, as you know with all the startups, this is the execution. The idea that you have a AI first email client seems like an obvious idea, but actually creating one, executing it, and then taking on Gmail is really difficult. Gmail, it's fascinating to see Google can't do it, because why? Because momentum, because it's hard to re-engineer the entirety of the system, because feels like Google is perfectly positioned to do it. Same with you have Perplexity, which I love, Google could technically take on Perplexity and do it much better, but they haven't, not yet. So, it's fascinating why that is for large companies, that is an advantage for little tech, they could be agile.<|eot_id|>"

# Split into user and assistant parts
parts = text.split("<|eot_id|>", 1)  # Split at first <|eot_id|>
user_text = parts[0] + "<|eot_id|>"  # User part includes <|eot_id|>
assistant_text = parts[1] if len(parts) > 1 else ""  # Assistant part

# Tokenize as a pair
tokenized = tokenizer(
    user_text,
    assistant_text,
    truncation=TruncationStrategy.ONLY_FIRST,  # Truncate user if needed
    max_length=256,
    padding="max_length",
    return_tensors="pt"
)

# Print results
print(f"Token count: {tokenized['input_ids'].shape[1]}")  # 256
print(tokenizer.decode(tokenized["input_ids"][0]))  # See the tokenized and truncated text