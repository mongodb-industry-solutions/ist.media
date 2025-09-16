#
# Add news sections to documents
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from openai import OpenAI
import os, keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news"
collection = client[dbName][collectionName]

ai = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

def detect_sections(text):
    prompt = f"""

    I want you to detect the sections of a news article. Each article
    belongs to at least one, and at most 3 sections. You decide!

    You can choose from these sections to build the list:

    Automotive
    Business
    Careers & Work
    Culture
    Education
    Entertainment
    Environment
    Finance
    Health
    Lifestyle
    Politics
    Real Estate
    Science & Tech
    Sports
    Travel

    Please use ONLY items from the list above. NO OTHER items!

    The list may not contain more than ONE of the following sections:
    - Finance
    - Business
    - Science & Tech

    Please return as a Python list without ``` or ```python that
    can be processed programmatically. The article starts now:

    {text}
    """
    try:
        response = ai.chat.completions.create(
            model = "grok-3",
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
    #for doc in collection.find():
    for doc in collection.find({ "sections" : { "$exists" : False }}):
        try:
            sections = detect_sections(doc['text'])
            print(sections + " " + doc['title'])
            collection.update_one({ "_id" : doc["_id"] },
                                  { "$set" : { "sections" : eval(sections) }})
        except Exception as e:
            print(e)

except Exception as e:
    print(e)
    exit(1)
