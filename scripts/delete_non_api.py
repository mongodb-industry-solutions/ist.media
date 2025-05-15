#
# clean up the news_incoming collection:
# - remove rss content
# - keep api content
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

# Delete documents where 'source' != 'api' or 'source' does not exist
result = collection.delete_many({
    "$or" : [
        { "source" : { "$ne" : "api" }},
        { "source" : { "$exists" : False }}
    ]
})

print(f"Deleted {result.deleted_count} documents.")
