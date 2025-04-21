#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from openai import OpenAI
import os, json, datetime, pymongo


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

MCONN = os.getenv('MONGODB_IST_MEDIA')
MBASE = "1_media_demo"

news_collection = pymongo.MongoClient(MCONN)[MBASE]["news"]
daily_collection = pymongo.MongoClient(MCONN)[MBASE]["daily"]

ai = OpenAI()


def get_mongodb_date_filter(natural_language_date):
    today = datetime.datetime.now(datetime.UTC)

    prompt = f"""
    Convert the following time expression into a MongoDB-compatible filter format.

    Example:
    - "last week" → {{ "$gte": "<YYYY-MM-DD>", "$lt": "<YYYY-MM-DD>" }}
    - "in January 2024" → {{ "$gte": "2024-01-01", "$lt": "2024-02-01" }}
    - "yesterday" → {{ "$gte": "<YYYY-MM-DD>", "$lt": "<YYYY-MM-DD>" }}

    Time expression: "{natural_language_date}"

    Never create date expressions that point to the future. Never do!

    If no conversion is possible, return the universal time filter that goes
    from January 2024 to "{today}".

    Return **only** a valid JSON object, without explanations or comments.
    No wrap of ```json.
    Today is "{today}"
    """

    response = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    # Ensure the response is valid JSON
    try:
        json_text = raw_text.strip("`")  # Remove possible markdown code block formatting
        date_filter = json.loads(json_text)
        return date_filter
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {raw_text}")
        return None  # Return None if parsing fails


def get_context():
    pipeline = [
        {
            "$match" : { "published" : get_mongodb_date_filter("today") }
        },
        {
            "$project" : { "uuid" : 1, "title" : 1, "text" : { "$substrCP" : [ "$text", 0, 1500 ]} }
        }
    ]
    results = list(news_collection.aggregate(pipeline))
    return results


prompt = f"""
You are a news reporting web service.

Generate a dossier of news from the text fields in the JSON list
given as the context. Please include insights from all JSON
documents provided.

Do not use bullet points or create a list of individual news
articles. Rather, combine them intelligently as appropriate and
create floating text.

Highlight and emphasize entities which are central to the story. For
instance, highlight and emphasize names of people, products,
companies.

Incorporate floating Links in the format
https://ist.media/post?uuid=<uuid> and use the title field of the
same JSON document for the link's text.

Each document has a unique uuid that must always be linked to its
respective title and text. When generating a reference or link,
always use uuid and title together. Do not mix UUIDs across
documents.

When generating links, always use the `uuid` from the same document
as the `title`.  Never invent or mix UUIDs.

Do not create a list of links, rather, include them in the floating
text where appropriate and where they match.

At the very beginning of the news dossier, add a comprehensive short
summary of headlines in this format: Headline 1 | Headline 2 | ...
Each Headline shall have a length between 20 and 60 characters.

"{ get_context() }"

"""

response = ai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)


now = datetime.datetime.now(datetime.UTC)
formatted_date = now.strftime("%d %B %Y")

summary = response.choices[0].message.content.strip()

daily_collection.update_one({ "day" : formatted_date },
                            { "$set" : { "summary" : summary }},
                            upsert=True)
