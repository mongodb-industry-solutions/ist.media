#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from openai import OpenAI
from datetime import datetime
from .agentic.main import user_prompt_prefix, ai_agent_compute_user_summary
import logging


logger = logging.getLogger(__name__)

def agentic_master_planner(app):
    """Master planner scheduled task. Runs inside the Flask app context."""
    from . import mongo

    with app.app_context():
        cursor = mongo.db.users.find(
            {
                "$or" : [
                    { "summary" : { "$exists" : False }},
                    { "$expr" : { "$gt" : [ "$last_active", "$summary.ts" ]}}
                ]
            },
            { "username" : 1 }
        )

        for user in cursor:
            username = user["username"]
            try:
                summary_text = ai_agent_compute_user_summary(username)
                mongo.db.users.update_one(
                    { "_id" : user["_id"] },
                    { "$set" : {
                        "summary" : {
                            "text" : summary_text,
                            "ts" : datetime.utcnow()
                        }
                    }}
                )
                logger.info("Summary regenerated for %s", username)
            except Exception as e:
                logger.error(str(e))
