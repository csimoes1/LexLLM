import json

# input_file = "transcripts_jsonl_trimmed/lex_episode_training_1.jsonl"
# output_file = "transcripts_jsonl_trimmed/lex_episode_training_1_formatted.jsonl"

# for loop that goes from i=2 to i=86
for i in range(7, 29, 7):
    input_file = f"transcripts_jsonl_trimmed/lex_episode_test_{i}.jsonl"
    output_file = f"transcripts_jsonl_trimmed/lex_episode_test_{i}_formatted.jsonl"

    print(f"Processing {input_file}...")

    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                data = json.loads(line.strip())
                dialog = data["dialog"]
                text_parts = ["<|begin_of_text|>"]
                for turn in dialog:
                    content = turn["content"] if turn["content"] else "<empty>"
                    role = "user" if turn["role"] == "user" else "assistant"  # Map your roles
                    prefix = f"<|start_header_id|>{role}<|end_header_id|>Lex: " if role == "user" else f"<|start_header_id|>{role}<|end_header_id|>Guest: "
                    text_parts.append(f"{prefix}{content}<|eot_id|>")
                text = "".join(text_parts)
                outfile.write(json.dumps({"text": text}) + "\n")
    except FileNotFoundError:
        print(f"File {input_file} not found, skipping...")

print(f"Formatted data saved to {output_file}")