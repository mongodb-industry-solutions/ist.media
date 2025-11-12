#
# Convert Daily News Summary into a Podcast
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

import os, datetime, time, requests
from bs4 import BeautifulSoup
from pydub import AudioSegment

audio_dir = '/usr/local/share/content/audio'
podcast_tmp_file = '/var/tmp/podcast_tmp.mp3'

token = os.getenv('AUTOCONTENT_API_KEY')
create_url = 'https://api.autocontentapi.com/Content/Create'
status_base_url = 'https://api.autocontentapi.com/content/status/'
poll_interval = 10
today = datetime.datetime.now(datetime.UTC)

response = requests.get("https://ist.media/feed")
response.raise_for_status()
soup = BeautifulSoup(response.content, "html.parser")
paragraphs = [p.get_text() for p in soup.body.find_all("p", recursive=False)]
news_content = "\n\n".join(paragraphs)

spoken_today = today.strftime("%B %-d") # e.g. September 7
request_data = {
    "resources": [ { "content": news_content, "type": "text" } ],
    "text": f"""
    Create a summary of the news for the day. Merge topics
    of similar category into one talk track. Explicitely mention
    the current date "{spoken_today}", so the audience knows of
    which day you talk and discuss news about. No social grunting,
    no "uh", "aha", "wow", "ok", "mhmm". Never! No intro with
    commenting how everything is interconnected. No mention of deepdive.
    """,
    "outputType": "audio"
}

def create_content():
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'accept': 'text/plain'
    }
    response = requests.post(create_url, json=request_data, headers=headers)
    response.raise_for_status()
    return response.json()

def poll_status(request_id):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }
    url = f'{status_base_url}{request_id}'

    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        status = data.get('status')
        error_message = data.get('error_message')
        audio_url = data.get('audio_url')
        audio_title = data.get('audio_title')

        if error_message:
            print('Error from status check:', error_message)
            return

        if status == 100:
            print('Content creation complete!')
            print('Audio URL:', audio_url)
            print('Audio Title:', audio_title)

            podcast_data = requests.get(audio_url).content
            with open(podcast_tmp_file, 'wb') as handler:
                handler.write(podcast_data)

            audio = AudioSegment.from_file(podcast_tmp_file, format="mp3")
            audio = audio.set_frame_rate(44100)
            filename = f"{audio_dir}/podcast-{today:%d.%m.%Y}.mp3"
            audio.export(filename, format="mp3", bitrate="192k")
            return

        print(f'Current status: {status}. Waiting for 100...')
        time.sleep(poll_interval)

def main():
    try:
        create_response = create_content()
        request_id = create_response.get('request_id')
        error_message = create_response.get('error_message')

        if error_message or not request_id:
            print('Error from create request:', error_message)
            return

        print('Request initiated. Request ID:', request_id)
        poll_status(request_id)

    except requests.HTTPError as http_err:
        print('HTTP error:', http_err)
    except Exception as err:
        print('Error:', err)

if __name__ == '__main__':
    main()
