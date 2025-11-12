#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from openai import OpenAI
from datetime import datetime
from .agentic.main import (
    ai_agent_compute_user_summary,
    ai_agent_compute_user_aquisition_promo )
import logging, ast

# TODO: In scheduler run, also
#         - change from state 'pending' to 'failed' after 4 hours,
#             if not 'success' in the meantime
#         - delete failed promotions after 2 days, to keep demo view clear

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
            { "username" : 1, "fullname" : 1 }
        )

        for user in cursor:
            username = user["username"]
            fullname = user["fullname"]

            # re-generate user summary for users with recent activity
            logger.info(f"(Re-)generating user summary for {username}")
            try:
                if not (summary_text := ai_agent_compute_user_summary(username)):
                    summary_text = "No insights available (yet)."
                mongo.db.users.update_one(
                    { "_id" : user["_id"] },
                    { "$set" : {
                        "summary" : {
                            "text" : summary_text,
                            "ts" : datetime.utcnow()
                        }
                    }}
                )
            except Exception as e:
                logger.error(f"While re-generating user summary: {e}")

            # for anonymous users with recent activity, check
            # - if there's already a promotion for them in place,
            # - if not, ask the ai agent to generate one, only if
            #   this is a suitable prospect. The agent will decide.
            if fullname != 'Anonymous User':
                return # acquisition promo only applies to anonymous users
            if not mongo.db.planner.find_one({ "username" : username }):
                if (result := ai_agent_compute_user_aquisition_promo(username)):
                    add_promotion, promo_type, promo_text, why = ast.literal_eval(result)
                    logger.info(f"User acquisition agent: {username}, {promo_type}, { promo_text}, {why}")
                    if add_promotion:
                        mongo.db.planner.update_one(
                            { "username" : username },
                            { "$set" : {
                                "promo" : {
                                    "text" : promo_text,
                                    "ts" : datetime.utcnow()
                                },
                                "type" : promo_type,
                                "why" : why
                            }},
                            upsert=True
                        )
