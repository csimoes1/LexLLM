import json

import requests
from bs4 import BeautifulSoup
from unidecode import unidecode

'''
Processes the transcript of a single episode of the Lex Fridman podcast.
This class should be called by LexScrape.py.
'''
class LexTranscriptProcessor:
    def __init__(self, episode_number: int, episode_url: str):
        self.episode_number = episode_number
        self.episode_url = episode_url
        self.transcript_data = []
        self.training_data = []
        self.userAsstPairs = []

    def load_and_parse_html(self):
        # Load the HTML content from the file
        # with open(f'Lex-Episode-{self.episode_number}.html', 'r', encoding='utf-8') as file:
        #     html_content = file.read()
        #     return BeautifulSoup(html_content, 'html.parser')

        try:
            response = requests.get(self.episode_url)
            response.raise_for_status()

            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        except requests.exceptions.RequestException as e:
            print(f"Error downloading transcript from {self.episode_url}: {e}")
            return None

    def extract_segments(self, soup):
        # Find all ts-segment divs
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
        # FORMAT:
        # {"dialog": [{"content":"...height of the empire state building","role":"user"},
        # {"content":"381 meters, or 1,250 feet, is the height of the Empire State Building. If you also account for the antenna, it brings up the total height to 443 meters, or 1,454 feet","role":"assistant"},
        # {"content":"...and need to know.\nSo what is the answer in feet?","role":"user"},
        # {"content":"1454 feet","role":"assistant"}]
        # }
        # must be format user (guest), assistant (Lex), user, assistant, etc.
        previous_text = ""
        previous_name_lex = False
        for data in self.transcript_data:
            current_text = unidecode(data['text'])
            # if previous_text is blank, then this is the first record just store previous_text and previous_name_lex
            # if previous_name_lex is Lex Fridman, then the previous_text was from the guest
            # if previous_name_lex is not Lex Fridman guest, then previous_text was of Lex
            if not previous_text:
                previous_text = current_text
                previous_name_lex = self.is_lex_fridman(data['name'])
                # if Lex is first to talk, put in blank user record
                if self.is_lex_fridman(data['name']):
                    self.training_data.append({'content': '', 'role': 'user'})

            else:
                # once we have a previous_name_lex wait until name changes and then output the previous_text
                if self.is_lex_fridman(data['name']) == previous_name_lex:
                    previous_text += ' ' + current_text
                else: # name has changed, so output previous_text
                    if previous_name_lex:
                        self.training_data.append({'content': previous_text, 'role': 'assistant'})
                    else:
                        # for user just use the last 15 words of the previous text
                        previous_text = ' '.join(previous_text.split()[-15:])
                        self.training_data.append({'content': previous_text, 'role': 'user'})

                    # reset previous_text and previous_name_lex
                    previous_name_lex = self.is_lex_fridman(data['name'])
                    previous_text = current_text

        # output the last record
        if previous_name_lex:
            self.training_data.append({'content': previous_text, 'role': 'assistant'})
        else:
            # for user just use the last 15 words of the previous text
            previous_text = ' '.join(previous_text.split()[-15:])
            self.training_data.append({'content': previous_text, 'role': 'user'})


    def create_training_data_chat(self):
        # FORMAT:
        # {"dialog": [{"content":"what is the height of the empire state building","role":"user"},
        # {"content":"381 meters, or 1,250 feet, is the height of the Empire State Building. If you also account for the antenna, it brings up the total height to 443 meters, or 1,454 feet","role":"assistant"},
        # {"content":"Some people need to pilot an aircraft above it and need to know.\nSo what is the answer in feet?","role":"user"},
        # {"content":"1454 feet","role":"assistant"}]
        # }
        # must be format user (guest), assistant (Lex), user, assistant, etc.
        previous_text = ""
        previous_name_lex = False
        for data in self.transcript_data:
            current_text = unidecode(data['text'])
            # if previous_text is blank, then this is the first record just store previous_text and previous_name_lex
            # if previous_name_lex is Lex Fridman, then the previous_text was from the guest
            # if previous_name_lex is not Lex Fridman guest, then previous_text was of Lex
            if not previous_text:
                previous_text = current_text
                previous_name_lex = self.is_lex_fridman(data['name'])
                # if Lex is first to talk, put in blank user record
                if self.is_lex_fridman(data['name']):
                    self.training_data.append({'content': '', 'role': 'user'})

            else:
                # once we have a previous_name_lex wait until name changes and then output the previous_text
                if self.is_lex_fridman(data['name']) == previous_name_lex:
                    previous_text += ' ' + current_text
                else: # name has changed, so output previous_text
                    if previous_name_lex:
                        self.training_data.append({'content': previous_text, 'role': 'assistant'})
                    else:
                        self.training_data.append({'content': previous_text, 'role': 'user'})

                    # reset previous_text and previous_name_lex
                    previous_name_lex = self.is_lex_fridman(data['name'])
                    previous_text = current_text

        # output the last record
        if previous_name_lex:
            self.training_data.append({'content': previous_text, 'role': 'assistant'})
        else:
            self.training_data.append({'content': previous_text, 'role': 'user'})


    def process_transcript(self):
        soup = self.load_and_parse_html()
        self.extract_segments(soup)
        self.create_training_data_chat_trimmed()
        # now divide training_data into user and assistant pairs and save to json
        for i in range(0, len(self.training_data), 2):
            if(i+1 < len(self.training_data)):
                self.userAsstPairs.append({'dialog': [self.training_data[i], self.training_data[i+1]]})
            else:
                # if the number of records is odd, then the last record is a user record
                self.userAsstPairs.append({'dialog': [self.training_data[i]]})

        print(f'Total lines in transcript_data: {len(self.transcript_data)}')
        print(f'Total lines in training_data: {len(self.training_data)}')
        print(f'Total lines in userAsstPairs: {len(self.userAsstPairs)}')
        return self.userAsstPairs

    def is_lex_fridman(self, name: str) -> bool:
        return name in ['Lex Fridman','Lex']


# main program
if __name__ == "__main__":
    processor = LexTranscriptProcessor(episode_number=1, episode_url='https://lexfridman.com/cursor-team-transcript')
    pairedOutput = processor.process_transcript()

    for line in pairedOutput:
        print(json.dumps(line) + '\n')


# Usage
# processor = LexTranscriptProcessor(episode_number=458)
# processor.process_transcript()