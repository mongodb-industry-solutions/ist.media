#
# Copyright (c) 2025 MongoDB Inc.
# Author: Benjamin Lorenz <benjamin.lorenz@mongodb.com>
#

from flask import render_template
from .. import mongo
from . import ai
from .main import ai_agent_compute_user_aquisition_promo


@ai.route("/acq/<id>")
def status(id):
    text = ai_agent_compute_user_aquisition_promo(id)
    return render_template('ai/status.html', text=text)


@ai.route("/planner")
def planner():
    docs = list(mongo.db.planner.find({}))
    return render_template('ai/planner.html', docs=docs)
