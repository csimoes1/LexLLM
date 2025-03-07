import requests
from bs4 import BeautifulSoup
import json
import os

from LexTranscriptProcessor import LexTranscriptProcessor

'''
Our MAIN class to scrape the Lex Fridman podcast transcripts and turn them into training, validation and test data.
'''

def get_transcript_links():
    """
    Fetches the podcast webpage and extracts links to episode pages that have transcripts.

    Args:
        podcast_page_url (str): URL of the main podcast page.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'episode_title' (str): Title of the episode.
              - 'transcript_url' (str): URL to the transcript page (if available).
              - 'episode_page_url' (str): URL to the episode page.
    """

    try:
        html_content = ""
        # open file "Lex Fridman Podcast - Episodes All.html" and read in the content
        with open("Lex Fridman Podcast - Episodes All.html", "r", encoding="utf-8") as file:
            html_content = file.read()

        # response = requests.get(podcast_page_url)
        # print(f"status_code= {response.status_code}")
        # response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(html_content, 'html.parser')

        episode_links_with_transcripts = []

        # Find all episode listings.  You'll need to inspect the webpage to find the correct CSS selectors.
        # Let's assume episodes are listed in articles with a class like 'episode-item' or similar.
        # Inspect the webpage source code to identify the correct container for episodes.
        episode_containers = soup.find_all('div', class_='guest') # You might need to refine this selector

        for episode_container in episode_containers:
            episode_title_element = episode_container.find('div', class_='vid-title') # Adjust selector as needed
            # if not episode_title_element:
            #     episode_title_element = episode_container.find('h3', class_='entry-title') # Trying h3 as fallback
            if not episode_title_element:
                print(f"No title found for episode container: {episode_container}")
                continue # Skip if no title found

            episode_title = episode_title_element.text.strip()

            episode_page_link_div = episode_container.find('div', class_='vid-materials')
            # find the anchor for the text "Transcript"
            episode_page_link_element = episode_page_link_div.find('a', string=lambda text: text and 'Transcript' in text)
            # episode_page_link_element = episode_title_element.find('a')
            if not episode_page_link_element:
                print(f"No transcript for episode: {episode_title}")
                continue # Skip if no episode page link

            episode_page_url = episode_page_link_element['href']
            print(f"episode_page_url= {episode_page_url}")
            # Check for transcript link within the episode container.
            # transcript_link_element = episode_container.find('a', href=lambda href: href and 'transcript' in href.lower())
            transcript_link_element = episode_page_url

            if episode_page_url:
                episode_links_with_transcripts.append({
                    'episode_title': episode_title,
                    'transcript_url': episode_page_url
                })
            else:
                print(f"No transcript link found for episode: {episode_title}")

        return episode_links_with_transcripts

    except requests.exceptions.RequestException as e:
        print(f"Error fetching podcast page: {e}")
        return []


def save_transcript_to_json(episode_data, transcript_text, output_dir="transcripts_json"):
    """
    Saves the transcript text and episode metadata to a JSON file.

    Args:
        episode_data (dict): Dictionary containing episode metadata (title, urls).
        transcript_text (str): The transcript text.
        output_dir (str): Directory to save JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sanitize episode title for filename (remove invalid characters)
    filename = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in episode_data['episode_title'])
    filename = filename.replace(" ", "_").lower() + ".json"
    filepath = os.path.join(output_dir, filename)

    json_data = {
        "episode_title": episode_data['episode_title'],
        "episode_page_url": episode_data['episode_page_url'],
        "transcript_url": episode_data['transcript_url'],
        "transcript_text": transcript_text
    }

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"Transcript for '{episode_data['episode_title']}' saved to: {filepath}")
    except Exception as e:
        print(f"Error saving JSON file for '{episode_data['episode_title']}': {e}")


if __name__ == "__main__":
    # podcast_url = "https://lexfridman.com/podcast"
    episodes_with_transcripts = get_transcript_links()

    for data in episodes_with_transcripts:
        print(data)

    if episodes_with_transcripts:

        training_data = []      # 73 episodes
        validation_data = []    # 9 episodes
        test_data = []          # 4 episodes

        print(f"Found {len(episodes_with_transcripts)} episodes with transcripts.")
        episode_number = 1
        filename = ""
        for episode_data in episodes_with_transcripts:
            print(f"\nProcessing episode: {episode_data['episode_title']}")
            processor = LexTranscriptProcessor(episode_number=episode_number, episode_url=episode_data['transcript_url'])
            pairedOutput = processor.process_transcript()
            print(f"Transcript text for episode {episode_number} complete")

            # randomly manually selects episodes for validation and testing
            if(episode_number in [5, 10, 15, 20, 25, 30, 35, 40, 45]):
                filename = f"transcripts_jsonl/lex_episode_validation_{episode_number}.jsonl"
            elif(episode_number in [7, 14, 21, 28]):
                filename = f"transcripts_jsonl/lex_episode_test_{episode_number}.jsonl"
            else:
                filename = f"transcripts_jsonl/lex_episode_training_{episode_number}.jsonl"

            with open(filename, 'w', encoding='utf-8') as f:
                for line in pairedOutput:
                    f.write(json.dumps(line) + '\n')

            episode_number += 1
            print(f"Finished processing episode: {episode_data['episode_title']}")

        print(f"training_data, validation_data, and test_data saved to json files")
    else:
        print("No episodes with transcripts found or an error occurred.")

    print(f"END OF PROGRAM")