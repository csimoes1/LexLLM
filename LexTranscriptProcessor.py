import json
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
from transformers import AutoTokenizer

class LexTranscriptProcessor:
    def __init__(self, episode_number: int, episode_url: str):
        self.episode_number = episode_number
        self.episode_url = episode_url
        self.transcript_data = []
        self.training_data = []
        self.userAsstPairs = []
        # Load tokenizer for LLaMA 3.2 3B Instruct
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_assistant_tokens = 200  # Maximum allowed tokens for assistant

    def load_and_parse_html(self):
        try:
            response = requests.get(self.episode_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except requests.exceptions.RequestException as e:
            print(f"Error downloading transcript from {self.episode_url}: {e}")
            return None

    def extract_segments(self, soup):
        segments = soup.find_all('div', class_='ts-segment')
        for segment in segments:
            name_element = segment.find('span', class_='ts-name')
            text_element = segment.find('span', class_='ts-text')
            if name_element and text_element:
                name = name_element.text.strip()
                text = text_element.text.strip()
                if not name and self.transcript_data:
                    self.transcript_data[-1]['text'] += ' ' + text
                else:
                    self.transcript_data.append({'name': name, 'text': text})

    def create_training_data_chat_trimmed(self):
        previous_text = ""
        previous_name_lex = False
        for data in self.transcript_data:
            current_text = unidecode(data['text'])
            if not previous_text:
                previous_text = current_text
                previous_name_lex = self.is_lex_fridman(data['name'])
                if self.is_lex_fridman(data['name']):
                    self.training_data.append("<|begin_of_text|><|start_header_id|>user<|end_header_id|><|introduction|><|eot_id|>")
            else:
                if self.is_lex_fridman(data['name']) == previous_name_lex:
                    previous_text += ' ' + current_text
                else:
                    if previous_name_lex:  # Assistant (Lex)
                        # Tokenize assistant text to check length
                        assistant_text = f"<|start_header_id|>assistant<|end_header_id|>{previous_text}<|eot_id|>"
                        tokens = self.tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
                        if len(tokens) <= self.max_assistant_tokens:
                            self.training_data.append(assistant_text)
                        else:
                            print(f"Skipping assistant response ({len(tokens)} tokens) exceeding {self.max_assistant_tokens} tokens")
                            self.training_data.pop()  # remove last user text we appended to training_data
                    else:  # User (Guest)
                        # previous_text = ' '.join(previous_text.split()[-100:]) # now letting tokenizer truncate this
                        self.training_data.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{previous_text}<|eot_id|>")

                    previous_name_lex = self.is_lex_fridman(data['name'])
                    previous_text = current_text

        # Handle the last record
        if previous_text:
            if previous_name_lex:  # Last is assistant
                assistant_text = f"<|start_header_id|>assistant<|end_header_id|>{previous_text}<|eot_id|>"
                tokens = self.tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
                if len(tokens) <= self.max_assistant_tokens:
                    self.training_data.append(assistant_text)
                else:
                    print(f"Skipping final assistant response ({len(tokens)} tokens) exceeding {self.max_assistant_tokens} tokens")
                    self.training_data.pop()  # remove last user text we appended to training_data
            # else:  # Last is user, just skip it
                # previous_text = ' '.join(previous_text.split()[-100:]) # now letting tokenizer truncate this
                # self.training_data.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{previous_text}<|eot_id|>")

    def process_transcript(self):
        soup = self.load_and_parse_html()
        if soup:
            self.extract_segments(soup)
            self.create_training_data_chat_trimmed()
            # Create user-assistant pairs
            for i in range(0, len(self.training_data), 2):
                if i + 1 < len(self.training_data):
                    self.userAsstPairs.append({'text': f"{self.training_data[i]}{self.training_data[i+1]}"})
                else:
                    # If odd number, keep the last user record
                    self.userAsstPairs.append({'text': self.training_data[i]})

            print(f'Total lines in transcript_data: {len(self.transcript_data)}')
            print(f'Total lines in training_data: {len(self.training_data)}')
            print(f'Total lines in userAsstPairs: {len(self.userAsstPairs)}')
        return self.userAsstPairs

    def is_lex_fridman(self, name: str) -> bool:
        return name in ['Lex Fridman', 'Lex']

if __name__ == "__main__":
    processor = LexTranscriptProcessor(episode_number=999, episode_url='https://lexfridman.com/deepseek-dylan-patel-nathan-lambert-transcript')
    pairedOutput = processor.process_transcript()

    for line in pairedOutput:
        print(json.dumps(line))