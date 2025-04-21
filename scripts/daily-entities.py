#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from openai import OpenAI
import os, re, json, datetime, pymongo


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
You are an entity extraction service.

Return a machine-readable Python list.

Generate a python list of entities from the text fields in the JSON
list given as the context. The list shall contain 15 items. The
entities shall only include names, locations, companies, concepts,
initiatives, events.

Return a machine-readable Python list. No ```python prefix.
No entities = prefix. Just pure [].

"{ get_context() }"

"""

response = ai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)


now = datetime.datetime.now(datetime.UTC)
formatted_date = now.strftime("%d %B %Y")

entities_string = response.choices[0].message.content.strip()
# handle unescaped quotes that might come back from the LLM (e.g. 'Goodman's Bay')
entities_string = re.sub(r"(?<=\w)'(?=\w)", "\\\\'", entities_string)
# convert to list of strings - can still fail if the LLM was f*cking it up
entities = eval(entities_string)
# remove keywords that are very long
entities = list(filter(lambda entity: len(entity) < 30, entities))
entities = entities[:20] # safety guard - sometimes OpenAI returns too much

daily_collection.update_one({ "day" : formatted_date },
                            { "$set" : { "entities" : entities }},
                            upsert=True)
