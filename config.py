#
# Copyright (c) 2024 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:

    # MongoDB Atlas
    MONGO_URI = os.environ['MONGODB_IST_MEDIA']

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    UPLOAD_FOLDER = '/home/bjjl/uploads'


class ProductionConfig(Config):
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
