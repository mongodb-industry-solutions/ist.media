#
# Copyright (c) 2021 Benjamin Lorenz
#

from flask import jsonify, request, current_app
from .. import mongo
from . import api
from .errors import ApiError, bad_request, unauthorized, forbidden
from bson import ObjectId
import datetime
import json
import pymongo



### some helper functions ###

def missing(inp):
    return ApiError(bad_request("missing `%s' attribute" % inp))

def get_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ApiError(bad_request('JSON: ' + e.msg))

### end helper functions ###



### extract JSON data ###

def get_string_attribute(data, key, max_len):
    try:
        attr_val = data[key]
    except:
        return "-"
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

### end extract JSON data ###



### the actual API functions start here ###

@api.route('/receive', methods=['GET'])
def receive():
    ip = request.environ.get("X-Real-IP", request.remote_addr)

    try:
        user_id = request.args.get('user_id')
        day = int(request.args.get('day'))
        month = int(request.args.get('month'))

    except:
        pass

    try:
        try: [day, month]
        except NameError:
            result = mongo.db.entries.find({ 'user_id' : user_id }) \
                                     .sort('timestamp', pymongo.DESCENDING) \
                                     .limit(10)
        else:
            expr = { '$expr' :
                     { '$and' : [
                         { '$eq' : [ '$user_id', user_id ] },
                         { '$eq' : [ { '$dayOfMonth' : '$timestamp' }, day ] },
                         { '$eq' : [ { '$month' : '$timestamp' }, month ] }
                     ]}
                    }
            result = mongo.db.entries.find(expr) \
                                     .sort('timestamp', pymongo.DESCENDING) \
                                     .limit(50)
        return jsonify(list(result))
    
    except Exception as e:
        return jsonify({ 'result' : 'ERROR', 'description' : str(e) })


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
