#
# Clone news articles to another database cluster
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import keyparams

client = MongoClient(keyparams.MONGO_URI)
clientNew = MongoClient(keyparams.MONGO_URI_NEW)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]
collectionNew = clientNew[dbName][collectionName]

i = 0
try:
    for doc in collection.find({ "thread.site" : "bnnbreaking.com" }):
        i += 1
        if i % 50 == 0:
            print("\n" + str(i))
        collectionNew.insert_one(doc)
        print(".", end="", flush=True)
except Exception as e:
    print(e)
    exit(1)

print("\nDone")
