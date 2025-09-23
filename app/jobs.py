#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

import logging, os
from pymongo import MongoClient

logger = logging.getLogger(__name__)

def agentic_master_planner():
    try:
        db = MongoClient(os.environ.get('MONGODB_IST_MEDIA'))["1_media_demo"]

        users = db["users"]
        engagement_events = db["engagement_events"]
        planner = db["planner"]

        planner.insert_one({ "foo" : 42 })

    except Exception as e:
        logger.error(str(e))
