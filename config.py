#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    # MongoDB Atlas
    MONGO_URI = os.environ['MONGODB_IST_MEDIA']

    # currently supported LLMs
    AVAILABLE_LLMS = { 'OpenAI GPT-3.5' : 'gpt-3.5-turbo',
                       'OpenAI GPT-4'   : 'gpt-4-turbo',
                       'OpenAI GPT-4o'  : 'gpt-4o' }

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):

    API_BASE_URL = 'http://localhost:9090/api'

    # turn Flask into debug mode
    DEBUG = True


class ProductionConfig(Config):

    API_BASE_URL = 'https://ist.media/api'

    # cookie security
    SESSION_COOKIE_SECURE    = True
    REMEMBER_COOKIE_SECURE   = True
    REMEMBER_COOKIE_HTTPONLY = True

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)


config = {
    'development' : DevelopmentConfig,
    'production'  : ProductionConfig
}
