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
    prompt = f"""
    Write three paragraphs in your own words based on the following text:

    {text}
    """

    try:
        response = ai.chat.completions.create(
            model = "gpt-4o",
            messages = [
                { "role" : "system", "content" : "You are a helpful assistant." },
                { "role" : "user", "content" : prompt }
            ],
            max_tokens = 2000,
            n = 1,
            temperature = 0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

try:
    for doc in incoming.find():

        if news.find_one({ 'uuid' : doc['uuid'] }):
            print("k", end="", flush=True)
            continue

        doc['text'] = gen(doc['text'])
        del doc['_id']

        try:
            news.insert_one(doc)
            print(".", end="", flush=True)
        except DuplicateKeyError:
            print("k", end="", flush=True)
        sleep(0.5)

except Exception as e:
    print(e)
    exit(1)

print("")
