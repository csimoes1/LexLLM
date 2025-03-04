import os


def load_files_with_prefix(directory, prefix, suffix="formatted.jsonl"):
    result = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            result.append(filename)
    # sort result by episode number
    result.sort(key=lambda x: int(x.split("_")[-2]))
    return result

# Load training and testing data
files = load_files_with_prefix("transcripts_jsonl_trimmed", "lex_episode_training_")
print(files)