#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import Flask
from flask_pymongo import PyMongo
from flask.json.provider import JSONProvider, DefaultJSONProvider
from bson import ObjectId
from config import config
import logging


mongo = PyMongo()
logger = logging.getLogger(__name__)

def json_default(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super(CustomJSONProvider, self).default(o)

def create_app(config_name):
    app = Flask(__name__)
    app.json = CustomJSONProvider(app)
    #app.json.ensure_ascii = False # don't escape unicode characters
    app.secret_key = 'IST_MEDIA_SECRET_KEY'
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    mongo.init_app(app)

    logging.basicConfig(
        filename = '/tmp/ist.media.log',
        level = logging.INFO,
        format = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app
