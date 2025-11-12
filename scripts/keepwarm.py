#
# Keep filesystem, network, and MongoDB caches warm
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
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0)
LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
TRANSPORT = httpx.HTTPTransport(retries=3)

ARTICLE_SELECTOR = "a.tm-post-link"
faker = Faker()

def get_articles(client, url):
    """Fetch and parse all article links from the given URL"""
    res = client.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.select(ARTICLE_SELECTOR)

def click_articles(client, links):
    """Click some article links with pause between each"""
    for i, a in enumerate(links[:2], 1):
        href = a.get("href")
        if not href:
            continue
        full_url = urljoin(BASE_URL, href)
        res = client.get(full_url)
        print(f"{i}0 {full_url} → {res.status_code}")
        # traversal of related posts
        article_soup = BeautifulSoup(res.content, 'html.parser')
        sidebar = article_soup.find('aside', class_='tm-aside-col')
        if not sidebar:
            print("  -> No sidebar found, skipping related posts search.")
            continue
        for j, a_tag in enumerate(sidebar.find_all('a',
                    href=lambda href: href and href.startswith('/post?uuid=')), 1):
            time.sleep(0.1)
            related_url = urljoin(BASE_URL, a_tag['href'])
            a_res = client.get(related_url)
            print(f"{i}{j} {related_url} → {a_res.status_code}")
        time.sleep(0.5)

def perform_search_and_click(client, query, traverse=True):
    """Submit search query and click resulting articles"""
    print(f"\n🔍 Performing search for: '{query}'", end=("\n" if traverse else ""))
    search_url = f"{BASE_URL}/?query={query.replace(' ', '+')}"
    links = get_articles(client, search_url)
    if traverse:
        print(f"Found {len(links)} articles")
        click_articles(client, links)

def generate_random_search_terms(n=10):
    """Generate n realistic search terms using Faker"""
    terms = []
    while len(terms) < n:
        term = faker.catch_phrase()
        if 3 <= len(term) <= 30:  # filter for reasonable length
            terms.append(term)
    return terms

# the main function
def keep_site_warm():

    with httpx.Client(headers=HEADERS,
                      timeout=TIMEOUT,
                      limits=LIMITS,
                      transport=TRANSPORT,
                      follow_redirects=True) as client:

        print("🔐 Step 1: Initial request (sets welcome cookie)...")
        try:
            r1 = client.get(BASE_URL)
            print("Status:", r1.status_code)
            print("🍪 Cookies after login:", client.cookies.jar)

            print("🔁 Step 2: Second visit to homepage (with cookie)...")
            homepage_links = get_articles(client, BASE_URL)
            print(f"Found {len(homepage_links)} homepage articles")
            click_articles(client, homepage_links)

            # Step 3: Iterate over random search results
            for term in generate_random_search_terms(5):
                perform_search_and_click(client, term, False)
                time.sleep(0.2)
            print("")
            for term in generate_random_search_terms(2):
                perform_search_and_click(client, term)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return

if __name__ == "__main__":
    keep_site_warm()
