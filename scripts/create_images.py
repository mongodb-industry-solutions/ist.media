#
# Generate images for the new articles
#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from time import sleep
from openai import OpenAI
import keyparams, requests

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news_incoming"
collection = client[dbName][collectionName]

tmp_dir = '/var/tmp/images.ist.media'
client = OpenAI()

i = 0
try:
    for doc in collection.find():
        i += 1
        if i % 50 == 0:
            print("\n" + str(i))

        long_prompt  = "Never show people, organizations, or copyrighted elements. " + doc['title'] + " " + doc['text'][:350]
        short_prompt = "Never show people, organizations, or copyrighted elements. " + doc['title']
        try:
            response = client.images.generate(
                model = "dall-e-3",
                prompt = long_prompt,
                size = "1792x1024",
                #quality = "hd",
                quality = "standard",
                n = 1,
            )
        except Exception:
            response = client.images.generate(
                model = "dall-e-3",
                prompt = short_prompt,
                size = "1792x1024",
                #quality = "hd",
                quality = "standard",
                n = 1,
            )
        img_data = requests.get(response.data[0].url).content
        with open(tmp_dir + '/' + doc['uuid'] + '.png', 'wb') as handler:
            handler.write(img_data)
        print(".", end="", flush=True)
        sleep(10) # let's be gentle - don't rush

except Exception as e:
    print(e)
    exit(1)

print("")
