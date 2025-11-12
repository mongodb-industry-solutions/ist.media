#
# Add vector embeddings to news documents
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from openai import OpenAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
import sys, keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news"
collection = client[dbName][collectionName]

openai = OpenAI(api_key=keyparams.openai_api_key)
def generate_openai_embeddings(text: str) -> list[float]:
    response = openai.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

process_all = len(sys.argv) > 1 and sys.argv[1].lower() == "all"

i = 0
try:
    if process_all:
        docs = collection.find()
    else:
        docs = collection.find({ "embedding" : { "$exists" : False }})

    for doc in docs:
        i += 1
        if i % 50 == 0:
            print("\n" + str(i))
        try:
            embedding = generate_openai_embeddings(doc['text'])
        except Exception as e:
            print("e", end="", flush=True)
            if not process_all:
                collection.delete_one({ "_id" : doc["_id"] })
            continue
        collection.update_one({ "_id" : doc["_id"] },
                              { "$set" : { "embedding" : embedding }})
        print(".", end="", flush=True)
except Exception as e:
    print(e)
    exit(1)

print("")
