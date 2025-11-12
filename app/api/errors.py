#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import jsonify
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
from . import api


def response(message, code):
    return jsonify({ 'error' : str(message) }), code


@api.app_errorhandler(BadRequest)
def bad_request(message):
    return response(message, 400)

@api.app_errorhandler(NotFound)
def not_found(message):
    return response(message, 404)

@api.app_errorhandler(InternalServerError)
def internal_server_error(message):
    suffix = " (this is most probably a bug - please report to benjamin.lorenz@mongodb.com)"
    return response(str(message) + suffix, 500)


class ApiError(Exception):
    def __init__(self, response):
        self.response = response
    def __str__(self):
        return repr(self.response)
