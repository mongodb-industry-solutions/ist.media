#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import current_app
import logging


logger = logging.getLogger(__name__)

def agentic_master_planner(app):
    """Master planner scheduled task. Runs inside the Flask app context."""
    from . import mongo

    with app.app_context():
        try:
            pass
            #mongo.db.planner.insert_one({ "test" : 42 })
        except Exception as e:
            logger.error(str(e))
