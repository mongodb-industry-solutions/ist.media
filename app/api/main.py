#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import jsonify, request, redirect
from flask import current_app as app
from .. import mongo, logger
from . import api
from .errors import ApiError, bad_request, internal_server_error
from bson import ObjectId
from openai import OpenAI
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os, re, datetime, json, geocoder, uuid


ip_info_cache_collection = mongo.db.ip_info_cache
news_incoming_collection = mongo.db.news_incoming

ai = OpenAI()

def debug(msg: str):
    if app.config['DEBUG']:
        print("[DEBUG]: " + msg)


def log(request):
    ip = request.environ.get("X-Real-IP", request.remote_addr)
    loc = ip_info_cache_collection.find_one({ "ip" : ip })
    if not loc:
        debug("Retrieving IP location information for " + ip + " with geocoder web service")
        ws = geocoder.ip(ip)
        loc = { "ip" : ip,
                "city" : ws.city if ws.city else " - ",
                "country" : ws.country if ws.country else " - " }
        ip_info_cache_collection.insert_one(loc)
    logger.info(ip + " (" + loc['city'] + ", " + loc['country'] + "): " + request.base_url +
            (" (" + str(request.content_length) + " bytes)" if request.content_length else ""))


def missing(inp):
    return ApiError(bad_request("missing `%s' attribute" % inp))


def get_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ApiError(bad_request('JSON: ' + e.msg))


def get_string_attribute(data, key, max_len):
    try:
        attr_val = data[key]
    except:
        raise ApiError(bad_request("missing `%s' attribute" % key))
    if type(attr_val) is not str:
        raise ApiError(bad_request("`%s' must be of type String" % key))
    len_val = len(attr_val)
    if len_val > max_len:
        raise ApiError(bad_request("`%s' size exceeded (%d > %d)" % (key, len_val, max_len)))
    return attr_val


def get_boolean_attribute(data, key):
    try:
        attr_val = data[key]
    except:
        return False
    if attr_val != "true" and attr_val != "false":
        raise ApiError(bad_request("`%s' must be of type Boolean (true or false)" % key))
    return attr_val == "true"



###
### Keyword Generation
###

def calculate_keywords(text: str, model_name: str) -> list[str]:
    if len(text) < 300:
        return [] # AI fails to generate meaningful keywords for input that is too short
    lcdocs = [ Document(page_content=text, metadata={ "source" : "local" }) ]
    prompt_template = """Return a machine-readable Python list.
    Given the context of the media article, please provide me with 6
    short keywords that capture the essence of the content and help
    finding the article while searching the web. Consider terms that
    are central to the article's subject and are likely to be imported
    for summarization. Please prioritize names of companies, names of
    persons, names of products, events, technical terms, business
    terms over general words. Return a machine-readable Python list.
    "{text}"
    KEYWORDS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name=model_name)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        keywords_string = stuff_chain.invoke(lcdocs)['output_text']
        debug('calculated keywords for "' + text[:25] + '...": ' + str(keywords_string))
        # handle unescaped quotes that might come back from the LLM (e.g. 'Goodman's Bay')
        keywords_string = re.sub(r"(?<=\w)'(?=\w)", "\\\\'", keywords_string)
        # convert to list of strings - can still fail if the LLM was f*cking it up
        keywords = eval(keywords_string)
        # remove keywords that are very long
        keywords = list(filter(lambda keyword: len(keyword) < 30, keywords))
        keywords = keywords[:7] # safety guard - sometimes OpenAI returns too much
    except Exception as e:
        print(str(e) + ": " + str(keywords_string)) # log file in /tmp
        keywords = []
    return keywords


@api.route('/keywords', methods=['POST'])
def keywords():
    log(request)
    # parameter handling
    try:
        data = get_data(request.data)
        text = get_string_attribute(data, 'text', 32768)
        llm = get_string_attribute(data, 'llm', 64)
    except ApiError as error:
        return error.response
    # real action starts here
    try:
        keywords = calculate_keywords(text, llm)
        return jsonify({ 'keywords' : keywords })
    except Exception as e:
        return ApiError(internal_server_error(str(e))).response


@api.route('/delete/<uuid>', methods=['DELETE'])
def delete(uuid):
    ip = request.environ.get("X-Real-IP", request.remote_addr)
    # to be implemented


@api.route('/create', methods=['POST'])
def create():
    log(request)
    # parameter handling
    try:
        data = get_data(request.data)
        title = get_string_attribute(data, 'title', 256)
        text = get_string_attribute(data, 'text', 32768)
        json_params = True
    except ApiError:
        title = request.form['title']
        text = request.form['text']
        json_params = False
    # real action starts here
    try:
        article = {
            'uuid' : str(uuid.uuid4()),  # Generate a random UUID
            'source' : 'api',
            'title' : title,
            'text' : text,
            'published' : datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        result = news_incoming_collection.insert_one(article)
        if result.inserted_id:
            if json_params:
                return jsonify({ 'status' : 'ok', 'id' : result.inserted_id })
            else:
                return redirect('/new')
        else:
            return jsonify({ 'status' : 'failed', 'message' : 'Failed to insert article' })
    except Exception as e:
        return jsonify({ 'status' : 'failed', 'message' : str(e) })


def fact_check_and_write_about(url):
    prompt = f"""

    Conduct a fact check on the news below and write your own article about it.
    No bullet points, at least 2500 characters in length, and write in English.

    {url}
    """
    try:
        response = ai.responses.create(
            model = "gpt-4o",
            tools = [ { "type" : "web_search_preview" } ],
            input = prompt
        )
        return response.output_text

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


def create_title(text):
    prompt = f"""

    Generate a title that reflects the essence of the article below, and
    that could be used as the headline of a newspaper's article.

    Provide a headline that does not mislead regarding timing - what happened
    in the past, what is current news, etc.

    Do not surround the headline with single or double quotes.

    {text}
    """
    try:
        response = ai.responses.create(
            model = "gpt-4o",
            input = prompt
        )
        return response.output_text

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


@api.route('/create_from_url', methods=['POST'])
def create_from_url():
    log(request)
    # parameter handling
    try:
        data = get_data(request.data)
        url = get_string_attribute(data, 'sourceurl', 512)
        json_params = True
    except ApiError:
        url = request.form['sourceurl']
        json_params = False
    # real action starts here

    text = fact_check_and_write_about(url)
    title = create_title(text)

    #print(title)
    #print(text)
    #return redirect('/newurl')

    try:
        article = {
            'uuid' : str(uuid.uuid4()),  # Generate a random UUID
            'source' : 'api',
            'title' : title,
            'text' : text,
            'published' : datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        result = news_incoming_collection.insert_one(article)
        if result.inserted_id:
            if json_params:
                return jsonify({ 'status' : 'ok', 'id' : result.inserted_id })
            else:
                return redirect('/newurl')
        else:
            return jsonify({ 'status' : 'failed', 'message' : 'Failed to insert article' })
    except Exception as e:
        return jsonify({ 'status' : 'failed', 'message' : str(e) })
