#
# Copyright (c) 2021 Benjamin Lorenz
#

from flask import jsonify
from app.exceptions import ValidationError
from . import api


def response(error, message, code):
    resp = jsonify({ 'result' : error, 'description' : message })
    resp.status_code = code
    return resp

def bad_request(message):
    return response('bad request', message, 400)

def unauthorized(message):
    return response('unauthorized', message, 401)

def forbidden(message):
    return response('forbidden', message, 403)


@api.errorhandler(ValidationError)
def validation_error(e):
    return bad_request(e.args[0])


class ApiError(Exception):
    def __init__(self, response):
        self.response = response
    def __str__(self):
        return repr(self.response)
