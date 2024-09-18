from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from openai import OpenAI
from time import sleep
import keyparams, os, requests, json, uuid

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
incoming = client[dbName]["news_incoming"]
news = client[dbName]["news"]

target_dir = '/var/tmp/images.ist.media'

ai = OpenAI()


def gen(text):
    prompt_title = f"Create a title of max 12 words, only capitalize the first word, never use surrounding quotes, for this text:\n\n{text}"
    prompt_text = f"Write five paragraphs in your own words based on the following text:\n\n{text}"
    try:
        response = ai.chat.completions.create(
            model = "gpt-4o",
            messages = [
                { "role" : "system", "content" : "You are a helpful assistant." },
                { "role" : "user", "content" : prompt_title }
            ],
            max_tokens = 2000,
            n = 1,
            temperature = 0.7
        )
        title = response.choices[0].message.content.strip()
        
        response = ai.chat.completions.create(
            model = "gpt-4o",
            messages = [
                { "role" : "system", "content" : "You are a helpful assistant." },
                { "role" : "user", "content" : prompt_text }
            ],
            max_tokens = 2000,
            n = 1,
            temperature = 0.7
        )
        text = response.choices[0].message.content.strip()

        return title, text
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

i = 0
try:
    for doc in incoming.find():
        i += 1
        if i % 50 == 0:
            print("\n" + str(i))
        title, text = gen(doc['text'])
        doc['title'] = title
        doc['text'] = text
        del doc['_id']
        news.insert_one(doc)
        print("-", end="", flush=True)
        sleep(0.5)
except Exception as e:
    print(e)
    exit(1)

print("")
