#
# Generate images for the new articles
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from time import sleep
from openai import AzureOpenAI
import keyparams, os, requests

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
incoming = client[dbName]["news_incoming"]
news = client[dbName]["news"]

tmp_dir = '/var/tmp/images.ist.media'

ai = AzureOpenAI(
    api_version="2024-02-01",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
)

i = 0
uuids_to_delete = []
try:
    for doc in incoming.find():

        if news.find_one({ 'uuid' : doc['uuid'] }):
            print("k", end="", flush=True)
            continue

        i += 1
        if i % 50 == 0:
            print("\n" + str(i))

        prompt = """Never show people, organizations, or copyrighted
        elements. If the following contains text that is not allowed by
        your safety system then please create a compliant image that
        comes close: """ + doc['title'] + " " + doc['text'][:350]

        try:
            response = ai.images.generate(
                model = "dall-e-3",
                prompt = prompt,
                size = "1792x1024",
                #quality = "hd",
                quality = "standard",
                n = 1)
        except:
            print("e", end="", flush=True)
            uuids_to_delete.append(doc['uuid'])
            continue
        img_data = requests.get(response.data[0].url).content
        with open(tmp_dir + '/' + doc['uuid'] + '.png', 'wb') as handler:
            handler.write(img_data)
        print(".", end="", flush=True)
        sleep(10) # let's be gentle - don't rush

    for uuid in uuids_to_delete:
        incoming.delete_one({ 'uuid' : uuid })
        print("d", end="", flush=True)

except Exception as e:
    print(e)
    exit(1)

print("")
