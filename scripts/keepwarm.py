#
# Keep the network connections, filesystem and MongoDB caches warm
# Run this script regularly from e.g. cron
#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>

import httpx
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from faker import Faker

BASE_URL = "https://ist.media"
AUTH = ("istmedia", "istmedia2024")
HEADERS = {"User-Agent": "Mozilla/5.0"}

FIXED_SEARCH_TERMS = ["iphone", "electric bike"]
ARTICLE_SELECTOR = "a.tm-post-link"
faker = Faker()

def get_articles(client, url):
    """Fetch and parse all article links from the given URL"""
    res = client.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.select(ARTICLE_SELECTOR)

def click_articles(client, links):
    """Click all article links with pause between each"""
    for i, a in enumerate(links[:4], 1):
        href = a.get("href")
        if not href:
            continue
        full_url = urljoin(BASE_URL, href)
        res = client.get(full_url)
        print(f"{i}. {full_url} → {res.status_code}")
        time.sleep(12)

def perform_search_and_click(client, query):
    """Submit search query and click resulting articles"""
    print(f"\n🔍 Performing search for: '{query}'")
    search_url = f"{BASE_URL}/?query={query.replace(' ', '+')}"
    links = get_articles(client, search_url)
    print(f"Found {len(links)} articles for '{query}'")
    click_articles(client, links)

def generate_random_search_terms(n=2):
    """Generate n realistic search terms using Faker"""
    terms = []
    while len(terms) < n:
        term = faker.catch_phrase()
        if 3 <= len(term) <= 30:  # filter for reasonable length
            terms.append(term)
    return terms

def keep_site_warm():
    with httpx.Client(auth=AUTH, headers=HEADERS, follow_redirects=True) as client:
        print("🔐 Step 1: Initial request (sets welcome cookie)...")
        r1 = client.get(BASE_URL)
        print("Status:", r1.status_code)

        print("🍪 Cookies after login:", client.cookies.jar)

        print("🔁 Step 2: Second visit to homepage (with cookie)...")
        homepage_links = get_articles(client, BASE_URL)
        print(f"Found {len(homepage_links)} homepage articles")
        click_articles(client, homepage_links)

        # Search with fixed terms
        #for term in FIXED_SEARCH_TERMS:
        #    perform_search_and_click(client, term)

        # Add 2 random terms using Faker
        random_terms = generate_random_search_terms()
        for term in random_terms:
            perform_search_and_click(client, term)

if __name__ == "__main__":
    keep_site_warm()
