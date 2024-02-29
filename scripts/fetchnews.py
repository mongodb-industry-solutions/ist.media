from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import time
import webzio
import keyparams

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "business_news"
collection = client[dbName][collectionName]

webzio.config(token=keyparams.webzio_token)
queryparams = {
    "q": "site_type:news category:\"Economy, Business and Finance\" num_chars:>300 language:english",
    "ts": str(int(time.time()) - 60*60*4),
    "sort": "crawled"
}
output = webzio.query("filterWebContent", queryparams)

while True:
    i = 0
    print("Next run starting...")
    for post in output['posts']:
        i = i+1
        try:
            collection.insert_one(post)
        except DuplicateKeyError:
            print("Duplicate found - skipping")
            exit(0)
    if i < 99:
       print("Done for now.")
       exit(0)
    print("Done inserting " + str(i) + " news items to MongoDB - sleeping a bit")
    time.sleep(2)
    output = webzio.get_next()
