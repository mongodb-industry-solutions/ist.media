#
# Collect articles from a news service
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from time import sleep
import keyparams, os, requests, json, uuid

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news_incoming"
collection = client[dbName][collectionName]

api_key = keyparams.NEWS_API_KEY
search_url = keyparams.NEWS_API_URL
tmp_dir = '/var/tmp/images.ist.media'

params = {
    'api-key' : api_key,
    'type' : 'article',
    'section' : 'world',
    'show-blocks' : 'all'
}

response = requests.get(search_url, params=params)
data = response.json()

if response.status_code == 200:
    collection.delete_many({}) # erase incoming
    for article in data['response']['results'][:5]:
        imageURL = None
        bodyHTML = ""
        rawText = ""
        article_id = article['id']
        webURL = article['webUrl']
        title = article['webTitle']
        publishedDate = article['webPublicationDate']
        elements = article['blocks']['main']['elements']
        for element in elements:
            if element['type'] == 'image':
                assets = element['assets']
                for asset in assets:
                    if asset['typeData']['width'] == 1000:
                        imageURL = asset['file']
                        break
        bodyChunks = article['blocks']['body']
        for chunk in bodyChunks:
            bodyHTML = chunk['bodyHtml']
            rawText = chunk['bodyTextSummary']
            break

        if imageURL:
            raw_article = {
                'uuid' : str(uuid.uuid5(uuid.NAMESPACE_DNS, article_id)),
                'weburl' : webURL,
                'title' : title,
                'published' : publishedDate,
                'imageurl' : imageURL,
                'html' : bodyHTML,
                'text' : rawText
            }
            try:
                collection.insert_one(raw_article)
            except DuplicateKeyError:
                print("d", end="", flush=True)
            except Exception:
                print("e")
                exit(1)
            else:
                print(".", end="", flush=True)
        else:
            print("x", end="", flush=True)
else:
    print(f"Error: {response.status_code} - {data}")

#try:
#    os.mkdir(tmp_dir)
#except Exception as e:
#    pass


#i = 0
#try:
#    for doc in collection.find():
#        i += 1
#        if i % 50 == 0:
#            print("\n" + str(i))
#        img_data = requests.get(doc['imageurl']).content
#        with open(tmp_dir + '/' + doc['uuid'] + '.png', 'wb') as handler:
#            handler.write(img_data)
#        print("-", end="", flush=True)
#        sleep(0.5)
#except Exception as e:
#    print(e)
#    exit(1)

print("")
