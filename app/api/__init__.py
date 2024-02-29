#
# Copyright (c) 2021 Benjamin Lorenz
#

from flask import Blueprint

api = Blueprint('api', __name__)

from . import main
