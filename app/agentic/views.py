#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import g, render_template, redirect, request
from flask import session, url_for, send_file, jsonify, current_app as app
from mistune import html
from .. import mongo, logger
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

@ai.route("/status", methods=["GET"])
def status():
    print("AI Status:")
    return render_template('ai/status.html')
