#
# Collect news from newsapi.ai
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from eventregistry import *
import time, json, os, sys, uuid, hashlib, keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]

er = EventRegistry(apiKey = "5f4327ac-4b71-45c6-bded-ad7f72c2eb72", allowUseOfArchive=False)


def create_uuid_from_string(val: str) -> str:
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))


try:
    q = QueryArticlesIter(keywords = QueryItems.OR([ "SEPA",
                                                     "Europe" ]),
                          sourceUri= er.getSourceUri("Seeking Alpha"),
                          lang="eng")
    output = q.execQuery(er, sortBy = "date", maxItems = 200)
except Exception as e:
    print(e)
    exit(1)


i = 0
t = 1
for item in output:
    if t % 50 == 0:
        print("\n" + str(t))
    try:
        post = {
            'uuid' : create_uuid_from_string(item['url']),
            'thread' : {
                'main_image' : item['image'],
                'site' : 'seekingalpha.com'
            },
            'title' : item['title'],
            'author' : 'Seeking Alpha',
            'published' : item['dateTimePub'],
            'text' : item['body']
        }
        collection.insert_one(post)
        i += 1
        t += 1
        print(".", end="", flush=True)
    except DuplicateKeyError:
        print("d", end="", flush=True)
        t += 1

print("\nNumber of articles inserted: " + str(i))
