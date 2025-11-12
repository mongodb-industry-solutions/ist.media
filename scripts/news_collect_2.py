#
# Collect articles from a news service
#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from time import sleep
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import keyparams, os, sys, requests, json, uuid, feedparser, cloudscraper

client = MongoClient(keyparams.MONGO_URI)
dbName = "1_media_demo"
collectionName = "news_incoming"
collection = client[dbName][collectionName]

try:
    sector_id = sys.argv[1]
except:
    print("No sector ID provided - exiting.")
    exit(1)

# Define the URL of the RSS feed
base_url = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01'
url = base_url + '&id=' + sector_id

# Parse the RSS feed
feed = feedparser.parse(url)

# Clean-up incoming collection
collection.delete_many({}) # erase incoming

# One instance of cloudscraper for all articles
scraper = cloudscraper.create_scraper()

# Function to extract full text from an article page
def get_full_text(article_url):
    try:
        # Fetch the article page
        response = scraper.get(article_url)
        status_code = response.status_code

        # Check if the request was successful
        if status_code == 200:
            # Parse the page content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all div elements with class 'group' and extract p tags inside them
            group_divs = soup.find_all('div', class_='group')

            # Initialize a variable to store the full text
            full_text = ''

            # Loop through each div with class 'group'
            for div in group_divs:
                # Find all p tags within the div and concatenate their text
                p_tags = div.find_all('p')
                for p in p_tags:
                    full_text += p.get_text().strip() + ' '

            return full_text.strip()
        else:
            print(
                f"Failed to fetch full article: {article_url}: HTTP status "
                f"code {status_code}"
            )
            return ""
    except Exception as e:
        print(f"Failed to fetch full article: {e}")
        return ""

# Loop through each article entry in the RSS feed
for entry in feed.entries[:5]:
    # Extract the title and link
    title = entry.title if 'title' in entry else ''
    article_url = entry.link if 'link' in entry else ''

    # Parse and format the published date as ISO 8601 for MongoDB
    date = entry.published if 'published' in entry else ''
    if date:
        try:
            date = date_parser.parse(date).strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception as e:
            date = None  # Handle date parsing errors

    # Fetch the full text from the article's page
    full_text = get_full_text(article_url) if article_url else ''

    if len(full_text) < 500: # skip articles without (sufficiently long) text body
        continue

    raw_article = {
        'uuid' : str(uuid.uuid5(uuid.NAMESPACE_DNS, entry.id)),
        'weburl' : article_url,
        'title' : title,
        'published' : date,
        'text' : full_text
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

print("")
