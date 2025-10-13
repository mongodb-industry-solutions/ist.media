#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import g, render_template, redirect, request
from flask import session, url_for, send_file, jsonify, current_app as app
from mistune import html
from .. import mongo, logger
from ..main.metrics import format_user_context_prompt
from . import ai
from typing import List, Dict, Any
from bson import Binary
from bson.objectid import ObjectId
from pymongo import ReturnDocument
from openai import OpenAI
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlencode
from requests.auth import HTTPBasicAuth
from json import JSONEncoder
import re, textwrap, string, os, json, time, uuid as python_uuid
import requests, geocoder, pycountry, math, voyageai
import io, qrcode, bcrypt, numpy as np


open_ai = OpenAI()
voyage_ai = voyageai.Client()


def user_prompt_prefix(username):
    data = mongo.db.users.find_one({ 'username' : username },
                                   { "engagement" : 1, "stats" : 1 })
    engagement = data['engagement']
    stats = data['stats']

    first_seen = datetime.fromisoformat(engagement['first_seen']).strftime('%B %d %Y')
    active_days = engagement['active_days_28']
    inactive_days = engagement['gaps_count_28']

    prompt_prefix = f"""

    You are an agent to invent and conduct user acquisition and retention
    experiments for the news website called ist.media. Here's information about
    the current user with username { username }.

    This user has been first seen on { first_seen } and was active on { active_days } days
    in the last 28 days, and with { inactive_days } days of no activity.

    Their current engagement indexes: { data['engagement']['smoothed_indexes'] }.
    These numbers are exponential moving averages and show the frequency of recent
    article consumption. The derived momentum is: { data['engagement']['momentum'] }.
    A value larger than 1 indicates increasing reading activity, a value lower
    than 1 indicates decreasing activity.

    { format_user_context_prompt(data['stats']) }

    """
    return prompt_prefix


def ai_agent_compute_user_summary(username):
    context = user_prompt_prefix(username)
    task = """Please provide a one-paragraph summary about the user.

    Mention their username. If the username looks like a hash key, it is an
    anonymous user, and you can expect that some of those users only show a
    reasonably short history of activity.

    Never show parantheses, and don't use technical terms,
    or variable names, or dict keys, to describe things, but rather explain
    for a non-technical user in business terms.
    """
    response = open_ai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            { "role" : "user", "content" : f"{context}\n\nTask: {task}" }
        ],
        max_tokens = 150,
        temperature = 0.7
    )
    return response.choices[0].message.content
