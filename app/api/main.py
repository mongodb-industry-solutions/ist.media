#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import jsonify, request, current_app
from .. import mongo
from . import api
from .errors import ApiError, bad_request, internal_server_error
from bson import ObjectId
from langchain.docstore.document import Document
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import datetime
import json
import pymongo


### some helper functions

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


#---------------------
### keyword generation

def calculate_keywords(text: str) -> list[str]:
    lcdocs = [ Document(page_content=text, metadata={"source": "local"}) ]
    prompt_template = """Return a machine-readable Python list.
    Given the context of the media article, please provide
    me with 6 short keywords that capture the essence of the content and help
    finding the article while searching the web. Consider terms
    that are central to the article's subject and are likely to be imported for
    summarization. Please prioritize names of companies, names of persons,
    names of products, events, technical terms, business terms
    over general words.
    Return a machine-readable Python list.
    "{text}"
    KEYWORDS:"""
    try:
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text")
        keywords_string = stuff_chain.invoke(lcdocs)['output_text']
        keywords = eval(keywords_string) # convert str into list
        keywords = list(filter(lambda keyword: len(keyword) < 30, keywords))
        keywords = keywords[:7] # safety guard - sometimes OpenAI returns too much
    except Exception as e:
        print(e) # will be printed in the log file that is residing in /tmp
        keywords = []
    return keywords


@api.route('/keywords', methods=['POST'])
def keywords():
    ip = request.environ.get("X-Real-IP", request.remote_addr)

    # parameter handling
    try:
        data = get_data(request.data)
        text = get_string_attribute(data, 'text', 32768)
    except ApiError as error:
        return error.response

    # real action starts here
    try:
        keywords = calculate_keywords(text)
        return jsonify({ 'keywords' : keywords })
    except Exception as e:
        return ApiError(internal_server_error(str(e))).response


@api.route('/send', methods=['POST'])
def send():
    ip = request.environ.get("X-Real-IP", request.remote_addr)
    
    try:
        data = get_data(request.data)
        user_id = get_string_attribute(data, 'user_id', 128)
        entry = get_string_attribute(data, 'entry', 8192)
        city = get_string_attribute(data, 'city', 256)
        is_encrypted = get_boolean_attribute(data, 'is_encrypted')
        
    except ApiError as error:
        return error.response

    if city == "-":
        doc = { 'user_id' : user_id, 'timestamp' : datetime.datetime.now(datetime.timezone.utc),
                'entry' : entry, 'is_encrypted' : is_encrypted }
    else:
        doc = { 'user_id' : user_id, 'timestamp' : datetime.datetime.now(datetime.timezone.utc),
                'entry' : entry, 'city' : city, 'is_encrypted' : is_encrypted }
    try:
        result = mongo.db.entries.insert_one(doc)
        return jsonify({ 'result' : 'OK', 'inserted' : str(result.inserted_id) })

    except Exception as e:
        return jsonify({ 'result' : 'ERROR', 'description' : str(e) })


@api.route('/delete/<object_id>', methods=['DELETE'])
def delete(object_id):
    ip = request.environ.get("X-Real-IP", request.remote_addr)

    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({ 'result' : 'ERROR', 'description' : 'user_id parameter missing' })

    try:
        result = mongo.db.entries.delete_one(
            { '_id' : ObjectId(object_id), 'user_id' : user_id })
        return jsonify({ 'result' : 'OK', 'deleted' : str(result.deleted_count) })

    except Exception as e:
        return jsonify({ 'result' : 'ERROR', 'description' : str(e) })
