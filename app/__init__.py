#
# Copyright (c) 2024, 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import Flask
from flask_pymongo import PyMongo
from flask.json.provider import JSONProvider, DefaultJSONProvider
from apscheduler.schedulers.background import BackgroundScheduler
from bson import ObjectId
from config import config
import logging


mongo = PyMongo()
mongo_preview = PyMongo()

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler(timezone="UTC")

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
    app.secret_key = 'IST_MEDIA_SECRET_KEY'
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    mongo.init_app(app)
    mongo_preview.init_app(app, uri=app.config["MONGO_URI_PREVIEW"])

    logging.basicConfig(
        filename = '/tmp/ist.media.log',
        level = app.config["LOG_LEVEL"],
        format = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')

    from .agentic import ai as ai_blueprint
    app.register_blueprint(ai_blueprint, url_prefix='/ai')

    # scheduler initialization
    from .jobs import agentic_master_planner

    if not scheduler.running:
        scheduler.start()

    scheduler.add_job(
        func = agentic_master_planner,
        args = [app],
        trigger = 'interval',
        seconds = 30,
        id = 'agentic master planner',
        max_instances = 1,
        replace_existing = True
    )

    return app
