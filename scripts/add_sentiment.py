#
# Add sentiment to news documents
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from openai import OpenAI
import keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news"
collection = client[dbName][collectionName]

ai = OpenAI()

def sentiment(text):
    prompt = f"""

    I want you to detect the sentiment of an article, which can be one
    of '+' (positive), 'o' (neutral), or '-' (negative).  Please
    return as a single char without quotes that can be processed
    programmatically. The article starts now:

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

i = 0
try:
    #for doc in collection.find():
    for doc in collection.find({ "sentiment" : { "$exists" : False }}):
        i += 1
        if i % 50 == 0:
            print("\n" + str(i))
        try:
            collection.update_one({ "_id" : doc["_id"] },
                                  { "$set" : { "sentiment" : sentiment(doc['text']) }})
            print(".", end="", flush=True)
        except:
            print("e", end="", flush=True)

except Exception as e:
    print(e)
    exit(1)

print("")
