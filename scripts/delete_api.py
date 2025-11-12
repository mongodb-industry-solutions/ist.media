#
# clean up the news_incoming collection:
# - remove api content
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import keyparams, os, requests, json, uuid

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news_incoming"
collection = client[dbName][collectionName]

result = collection.delete_many({ "source" : "api" })
print(f"Deleted {result.deleted_count} documents.")
