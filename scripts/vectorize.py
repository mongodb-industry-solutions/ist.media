#
# Add vector embeddings to news documents
#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]

try:
    for doc in collection.find({ "thread.site" : "bnnbreaking.com" }):
        print(doc["uuid"])
except Exception as e:
    print(e)
    exit(1)
